"""
Ensemble forecasting model: ARIMA + XGBoost with MLflow tracking.
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "sales-forecasting"


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100


class EnsembleForecaster:
    def __init__(self, arima_order=(2, 1, 2), xgb_params=None, arima_weight=0.35):
        self.arima_order = arima_order
        self.arima_weight = arima_weight
        self.xgb_weight = 1 - arima_weight
        self.xgb_params = xgb_params or {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
        self.arima_model = None
        self.xgb_model = None

    def fit(self, train_df: pd.DataFrame, target_col: str = "sales",
            feature_cols: list = None, run_name: str = "ensemble_run"):
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "arima_order": str(self.arima_order),
                "arima_weight": self.arima_weight,
                **{f"xgb_{k}": v for k, v in self.xgb_params.items()}
            })
            
            # ARIMA on raw series
            self.arima_model = ARIMA(train_df[target_col], order=self.arima_order)
            self.arima_fitted = self.arima_model.fit()
            
            # XGBoost on engineered features
            if feature_cols is None:
                feature_cols = [c for c in train_df.columns if c not in [target_col, "date"]]
            
            X_train = train_df[feature_cols].fillna(0)
            y_train = train_df[target_col]
            
            self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
            self.xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
            self.feature_cols = feature_cols
            
            # Log training metrics
            train_pred = self.predict_in_sample(train_df, target_col)
            metrics = {
                "train_mape": mape(y_train.values, train_pred),
                "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
                "train_mae": mean_absolute_error(y_train, train_pred)
            }
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(self.xgb_model, "xgb_model")
            print(f"Training complete | MAPE: {metrics['train_mape']:.2f}% | RMSE: {metrics['train_rmse']:.0f}")
        
        return self

    def predict_in_sample(self, df, target_col="sales"):
        arima_pred = self.arima_fitted.fittedvalues.values
        xgb_pred = self.xgb_model.predict(df[self.feature_cols].fillna(0))
        n = min(len(arima_pred), len(xgb_pred))
        return self.arima_weight * arima_pred[:n] + self.xgb_weight * xgb_pred[:n]

    def forecast(self, steps: int = 30) -> np.ndarray:
        arima_fc = self.arima_fitted.forecast(steps=steps)
        # XGBoost forecast requires feature projection (simplified here)
        xgb_fc = np.array([self.xgb_model.predict([[i] + [0] * (len(self.feature_cols) - 1)])[0]
                           for i in range(steps)])
        return self.arima_weight * arima_fc + self.xgb_weight * xgb_fc
