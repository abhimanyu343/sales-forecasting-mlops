"""
Ensemble model combining SARIMA, LightGBM, and XGBoost forecasts.

Ensemble strategy:
1. Train each base model independently with MLflow tracking
2. Generate out-of-fold (OOF) predictions on validation set
3. Compute optimal weights from OOF MAPE using constrained optimisation
4. Final prediction: weighted average of base model forecasts
5. Isotonic regression calibration on residuals (optional)

The ensemble is serialised as a single pickle for serving.
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
import pickle
import logging
import time
from typing import List, Dict, Optional, Tuple, Any
from scipy.optimize import minimize
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "sales-forecasting-ensemble"


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def find_optimal_weights(
    oof_preds: Dict[str, np.ndarray],
    y_true: np.ndarray
) -> Dict[str, float]:
    """
    Find model weights that minimise ensemble MAPE via constrained optimisation.
    Weights are constrained to be non-negative and sum to 1.

    Args:
        oof_preds: dict mapping model_name → OOF prediction array
        y_true: True values aligned with OOF predictions

    Returns:
        dict mapping model_name → optimal weight
    """
    model_names = list(oof_preds.keys())
    pred_matrix = np.column_stack([oof_preds[m] for m in model_names])

    def neg_mape(weights):
        ensemble_pred = pred_matrix @ weights
        return mape(y_true, ensemble_pred)

    n = len(model_names)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(neg_mape, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-8})

    weights = result.x.clip(0)
    weights /= weights.sum()  # re-normalise after clipping

    weight_dict = dict(zip(model_names, weights.round(4)))
    log.info(f"Optimal ensemble weights: {weight_dict} | Ensemble MAPE: {result.fun:.2f}%")
    return weight_dict


class SalesEnsemble:
    """
    Weighted ensemble of SARIMA + LightGBM + XGBoost for sales forecasting.

    Attributes:
        weights: Per-model forecast weights (optimised from validation MAPE)
        base_models: Dict of fitted base models
        feature_cols: Feature columns used by gradient boosting models
        metrics: Training and validation performance metrics
    """

    def __init__(self, run_name: str = "ensemble"):
        self.weights: Dict[str, float] = {}
        self.base_models: Dict[str, Any] = {}
        self.feature_cols: List[str] = []
        self.target_col: str = "sales"
        self.date_col: str = "date"
        self.metrics: Dict[str, Any] = {}
        self.run_name = run_name
        self.is_fitted = False

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "sales",
        date_col: str = "date",
    ) -> "SalesEnsemble":
        """
        Train all base models and compute ensemble weights.

        Args:
            train_df: Training DataFrame (features + target + date)
            val_df:   Validation DataFrame for weight optimisation
            feature_cols: Feature columns for gradient boosting models
            target_col: Target column name
            date_col: Date column name
        """
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.date_col = date_col

        oof_preds: Dict[str, np.ndarray] = {}
        y_val = val_df[target_col].values

        with mlflow.start_run(run_name=self.run_name) as run:
            self._run_id = run.info.run_id

            # ── SARIMA ────────────────────────────────────────────────────────
            log.info("Training SARIMA...")
            t0 = time.time()
            sarima_pred = self._fit_sarima(train_df, val_df, target_col)
            sarima_time = time.time() - t0
            oof_preds["sarima"] = sarima_pred
            sarima_mape = mape(y_val, sarima_pred)
            mlflow.log_metrics({"sarima_val_mape": sarima_mape,
                                "sarima_val_rmse": rmse(y_val, sarima_pred),
                                "sarima_train_time_s": sarima_time})
            log.info(f"SARIMA | MAPE={sarima_mape:.2f}% in {sarima_time:.1f}s")

            # ── LightGBM ─────────────────────────────────────────────────────
            log.info("Training LightGBM...")
            t0 = time.time()
            lgbm_pred = self._fit_lightgbm(train_df, val_df, feature_cols, target_col)
            lgbm_time = time.time() - t0
            oof_preds["lightgbm"] = lgbm_pred
            lgbm_mape = mape(y_val, lgbm_pred)
            mlflow.log_metrics({"lgbm_val_mape": lgbm_mape,
                                "lgbm_val_rmse": rmse(y_val, lgbm_pred),
                                "lgbm_train_time_s": lgbm_time})
            log.info(f"LightGBM | MAPE={lgbm_mape:.2f}% in {lgbm_time:.1f}s")

            # ── XGBoost ──────────────────────────────────────────────────────
            log.info("Training XGBoost...")
            t0 = time.time()
            xgb_pred = self._fit_xgboost(train_df, val_df, feature_cols, target_col)
            xgb_time = time.time() - t0
            oof_preds["xgboost"] = xgb_pred
            xgb_mape = mape(y_val, xgb_pred)
            mlflow.log_metrics({"xgb_val_mape": xgb_mape,
                                "xgb_val_rmse": rmse(y_val, xgb_pred),
                                "xgb_train_time_s": xgb_time})
            log.info(f"XGBoost | MAPE={xgb_mape:.2f}% in {xgb_time:.1f}s")

            # ── Optimise weights ──────────────────────────────────────────────
            self.weights = find_optimal_weights(oof_preds, y_val)
            ensemble_pred = sum(self.weights[m] * oof_preds[m] for m in oof_preds)
            ensemble_mape_val = mape(y_val, ensemble_pred)
            ensemble_rmse_val = rmse(y_val, ensemble_pred)
            ensemble_mae_val  = mae(y_val, ensemble_pred)

            self.metrics = {
                "ensemble_val_mape": ensemble_mape_val,
                "ensemble_val_rmse": ensemble_rmse_val,
                "ensemble_val_mae":  ensemble_mae_val,
                "weights": self.weights,
                "base_mape": {"sarima": sarima_mape, "lightgbm": lgbm_mape, "xgboost": xgb_mape}
            }
            mlflow.log_metrics({
                "ensemble_val_mape": ensemble_mape_val,
                "ensemble_val_rmse": ensemble_rmse_val,
                "ensemble_val_mae":  ensemble_mae_val,
            })
            mlflow.log_params({f"weight_{m}": round(w, 4) for m, w in self.weights.items()})

            log.info(f"Ensemble | MAPE={ensemble_mape_val:.2f}% RMSE={ensemble_rmse_val:.0f}")

        self.is_fitted = True
        return self

    def _fit_sarima(self, train_df, val_df, target_col) -> np.ndarray:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.stattools import adfuller
        import itertools

        series = train_df.set_index(self.date_col)[target_col]

        # Auto-select SARIMA order by AIC over small search space
        best_aic, best_order, best_seasonal = np.inf, (1, 1, 1), (1, 1, 1, 7)
        for p, d, q in itertools.product([0, 1, 2], [0, 1], [0, 1, 2]):
            try:
                m = SARIMAX(series, order=(p, d, q), seasonal_order=(1, 1, 1, 7),
                            enforce_stationarity=False, enforce_invertibility=False)
                res = m.fit(disp=False, maxiter=50)
                if res.aic < best_aic:
                    best_aic, best_order = res.aic, (p, d, q)
            except Exception:
                pass

        model = SARIMAX(series, order=best_order, seasonal_order=best_seasonal,
                        enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False)
        self.base_models["sarima"] = fitted

        forecast = fitted.forecast(steps=len(val_df))
        return np.maximum(forecast.values, 0)

    def _fit_lightgbm(self, train_df, val_df, feature_cols, target_col) -> np.ndarray:
        import lightgbm as lgb

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col].values
        X_val   = val_df[feature_cols].fillna(0)
        y_val   = val_df[target_col].values

        params = {
            "objective": "regression_l1",  # MAE — more robust to outliers
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)]
        )
        self.base_models["lightgbm"] = model
        return np.maximum(model.predict(X_val), 0)

    def _fit_xgboost(self, train_df, val_df, feature_cols, target_col) -> np.ndarray:
        import xgboost as xgb

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col].values
        X_val   = val_df[feature_cols].fillna(0)

        params = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.1,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
            "objective": "reg:absoluteerror",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, val_df[target_col].values)],
                  verbose=False)
        self.base_models["xgboost"] = model
        return np.maximum(model.predict(X_val), 0)

    def predict(self, df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
        """
        Generate ensemble forecast for given horizon.

        Args:
            df: Feature DataFrame (must include same features as training)
            horizon: Number of steps ahead to forecast

        Returns:
            DataFrame with columns: date, forecast, lower_80, upper_80
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        preds = {}
        if "sarima" in self.base_models:
            preds["sarima"] = np.maximum(self.base_models["sarima"].forecast(horizon), 0)
        if "lightgbm" in self.base_models:
            X = df.tail(horizon)[self.feature_cols].fillna(0)
            preds["lightgbm"] = np.maximum(self.base_models["lightgbm"].predict(X), 0)
        if "xgboost" in self.base_models:
            X = df.tail(horizon)[self.feature_cols].fillna(0)
            preds["xgboost"] = np.maximum(self.base_models["xgboost"].predict(X), 0)

        # Weighted ensemble
        available_weights = {m: self.weights.get(m, 0) for m in preds}
        total_w = sum(available_weights.values())
        ensemble = sum((available_weights[m] / total_w) * preds[m] for m in preds)

        # Approximate prediction intervals from SARIMA (if available)
        if "sarima" in self.base_models:
            forecast_result = self.base_models["sarima"].get_forecast(horizon)
            ci_80 = forecast_result.conf_int(alpha=0.2)
            lower = np.maximum(ci_80.iloc[:, 0].values, 0)
            upper = ci_80.iloc[:, 1].values
        else:
            # Simple ±15% as fallback
            lower = ensemble * 0.85
            upper = ensemble * 1.15

        dates = pd.date_range(
            start=df[self.date_col].max() + pd.Timedelta(days=1),
            periods=horizon, freq="D"
        )
        return pd.DataFrame({
            "date": dates,
            "forecast": ensemble.round(2),
            "lower_80": lower.round(2),
            "upper_80": upper.round(2),
        })

    def save(self, path: str = "models/ensemble.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: str = "models/ensemble.pkl") -> "SalesEnsemble":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        log.info(f"Ensemble loaded from {path}")
        return obj
