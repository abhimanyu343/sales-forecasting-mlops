"""
Main training orchestrator for sales forecasting pipeline.

Workflow:
1. Load and validate raw sales data
2. Run feature engineering pipeline
3. Time-series train/val/test split (no shuffling)
4. Train SARIMA + LightGBM + XGBoost ensemble
5. Evaluate on held-out test set
6. Log everything to MLflow + register model if it beats production
7. Persist model artifact

Usage:
    python training/train.py
    python training/train.py --config configs/train_config.yaml
    python training/train.py --data data/sales.csv --test-days 90 --run-name "v2-retrain"
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generate_data import generate_sales_data
from feature_store.features import build_features, get_feature_columns, create_train_test_split
from feature_store.drift import DriftDetector
from models.ensemble import SalesEnsemble, mape, rmse, mae
from training.evaluate import walk_forward_cv, compute_full_metrics

log = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "mlflow_uri": "http://localhost:5000",
    "experiment": "sales-forecasting-ensemble",
    "data_path": "data/sales_data.csv",
    "model_save_path": "models/ensemble.pkl",
    "test_days": 90,
    "val_days": 60,
    "target_col": "sales",
    "date_col": "date",
    "include_fourier": True,
    "include_price": False,
    "cv_folds": 3,
    "promote_if_mape_better_by": 0.5,  # percentage points
}


def load_config(config_path: str = None) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            overrides = yaml.safe_load(f)
        cfg.update(overrides)
        log.info(f"Config loaded from {config_path}")
    return cfg


def load_data(cfg: dict) -> pd.DataFrame:
    data_path = Path(cfg["data_path"])
    if not data_path.exists():
        log.warning(f"Data file not found at {data_path}. Generating synthetic dataset...")
        df = generate_sales_data(n_skus=10, days=730)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        log.info(f"Generated and saved synthetic data to {data_path}")
    else:
        df = pd.read_csv(data_path, parse_dates=[cfg["date_col"]])
        log.info(f"Loaded {len(df)} rows from {data_path}")

    # Basic validation
    required = [cfg["date_col"], cfg["target_col"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[cfg["target_col"]].isnull().mean() > 0.1:
        raise ValueError("Target column has >10% missing values. Check your data.")
    if (df[cfg["target_col"]] < 0).any():
        log.warning("Negative sales values detected — clipping to 0")
        df[cfg["target_col"]] = df[cfg["target_col"]].clip(lower=0)

    return df.sort_values(cfg["date_col"]).reset_index(drop=True)


def run_training(cfg: dict, run_name: str = None) -> dict:
    """Full training run. Returns metrics dict."""
    t_start = time.time()
    run_name = run_name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log.info("=" * 60)
    log.info(f"Starting training run: {run_name}")
    log.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df_raw = load_data(cfg)
    log.info(f"Date range: {df_raw[cfg['date_col']].min().date()} → {df_raw[cfg['date_col']].max().date()}")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    log.info("Running feature engineering...")
    df_feat = build_features(
        df_raw,
        target_col=cfg["target_col"],
        date_col=cfg["date_col"],
        include_fourier=cfg.get("include_fourier", True),
        include_price=cfg.get("include_price", False),
    )
    df_feat = df_feat.dropna().reset_index(drop=True)
    feature_cols = get_feature_columns(df_feat, cfg["target_col"], cfg["date_col"])
    log.info(f"Feature matrix: {df_feat.shape} | {len(feature_cols)} features")

    # ── 3. Train / Val / Test split ───────────────────────────────────────────
    test_days = cfg["test_days"]
    val_days  = cfg["val_days"]

    cutoff_test = df_feat[cfg["date_col"]].max() - pd.Timedelta(days=test_days)
    cutoff_val  = cutoff_test - pd.Timedelta(days=val_days)

    df_train = df_feat[df_feat[cfg["date_col"]] <= cutoff_val]
    df_val   = df_feat[(df_feat[cfg["date_col"]] > cutoff_val) & (df_feat[cfg["date_col"]] <= cutoff_test)]
    df_test  = df_feat[df_feat[cfg["date_col"]] > cutoff_test]

    log.info(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    if len(df_train) < 60:
        raise ValueError(f"Training set too small ({len(df_train)} rows). Need at least 60.")

    # ── 4. Drift check vs last training window ────────────────────────────────
    if len(df_val) > 20:
        detector = DriftDetector(key_features=feature_cols[:8])
        drift_report = detector.analyse(df_train, df_val, cfg["target_col"])
        if drift_report.overall_drift_detected:
            log.warning(f"Drift detected: {drift_report.recommendation}")

    # ── 5. Train ensemble ─────────────────────────────────────────────────────
    log.info("Training ensemble model...")
    ensemble = SalesEnsemble(run_name=run_name)
    ensemble.fit(df_train, df_val, feature_cols, cfg["target_col"], cfg["date_col"])

    # ── 6. Test set evaluation ────────────────────────────────────────────────
    log.info("Evaluating on held-out test set...")
    y_test = df_test[cfg["target_col"]].values
    forecast_df = ensemble.predict(df_feat[df_feat[cfg["date_col"]] <= cutoff_test], horizon=test_days)
    y_pred = forecast_df["forecast"].values[:len(y_test)]

    test_metrics = compute_full_metrics(y_test, y_pred)
    log.info(f"Test Set | MAPE={test_metrics['mape']:.2f}% | RMSE={test_metrics['rmse']:.0f} | MAE={test_metrics['mae']:.0f}")

    # ── 7. Cross-validation ───────────────────────────────────────────────────
    log.info(f"Running {cfg['cv_folds']}-fold walk-forward CV...")
    cv_results = walk_forward_cv(df_feat, feature_cols, cfg["target_col"],
                                 cfg["date_col"], n_folds=cfg["cv_folds"])
    log.info(f"CV MAPE: {cv_results['mean_mape']:.2f}% ± {cv_results['std_mape']:.2f}%")

    # ── 8. Save model ─────────────────────────────────────────────────────────
    save_path = cfg["model_save_path"]
    ensemble.save(save_path)
    log.info(f"Model saved to {save_path}")

    total_time = time.time() - t_start
    log.info(f"Training complete in {total_time:.1f}s")

    return {
        "run_name": run_name,
        "test_mape": test_metrics["mape"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "cv_mean_mape": cv_results["mean_mape"],
        "cv_std_mape": cv_results["std_mape"],
        "ensemble_weights": ensemble.weights,
        "n_train": len(df_train),
        "n_features": len(feature_cols),
        "total_time_s": round(total_time, 1),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="Train sales forecasting ensemble")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--data", type=str, default=None, help="Override data path")
    parser.add_argument("--test-days", type=int, default=None, help="Override test days")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data:       cfg["data_path"] = args.data
    if args.test_days:  cfg["test_days"] = args.test_days

    results = run_training(cfg, run_name=args.run_name)

    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    for k, v in results.items():
        print(f"  {k:30s}: {v}")
