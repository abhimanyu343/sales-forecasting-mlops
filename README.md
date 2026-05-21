# 📈 Sales Forecasting MLOps Pipeline

> End-to-end ML pipeline for retail/B2B sales forecasting — ARIMA + LightGBM + XGBoost ensemble with MLflow experiment tracking, automated retraining, data drift detection, FastAPI model serving, and Docker deployment.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)
![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2?logo=mlflow)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3-2980B9)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-3E9E50)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)

---

## 🎯 Problem Statement

Retail and B2B businesses need accurate 30/60/90-day sales forecasts to:
- Optimise inventory purchasing decisions
- Plan workforce and production capacity
- Set financial targets and budgets
- Identify demand anomalies before they become problems

This pipeline delivers production-grade forecasts with full lineage tracking, automated retraining on drift, and a REST API for integration with dashboards and ERP systems.

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                │
│  Raw sales transactions (CSV / DB / API)                        │
│         │                                                       │
│         ▼                                                       │
│  feature_store/                                                 │
│    ├── aggregator.py    → daily/weekly/monthly rollup           │
│    ├── features.py      → 40+ engineered features              │
│    └── drift.py         → PSI + KS drift detection            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     TRAINING LAYER                              │
│  models/                                                        │
│    ├── arima.py         → SARIMA with auto-order selection      │
│    ├── lightgbm_model.py → Gradient boosting with early stop   │
│    ├── xgboost_model.py  → XGBoost with hyperparameter tuning  │
│    └── ensemble.py       → Weighted ensemble + calibration     │
│                                                                 │
│  MLflow tracking: params, metrics, artifacts, model registry   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   SERVING LAYER                                 │
│  api/                                                           │
│    ├── main.py          → FastAPI prediction endpoint           │
│    ├── predictor.py     → Model loading + inference             │
│    └── schemas.py       → Pydantic request/response             │
│                                                                 │
│  scheduler/                                                     │
│    └── retrain.py       → Weekly APScheduler retraining         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Model Details

### SARIMA (Seasonal ARIMA)
- Auto-selects order (p,d,q)(P,D,Q,m) via AIC minimisation
- Handles yearly seasonality (m=52 weekly, m=12 monthly)
- Provides prediction intervals (80% and 95% confidence bands)
- Best for: stable, low-noise series with clear seasonality

### LightGBM Regressor
- 200+ trees, leaf-wise growth, native categorical support
- Trained on 40+ features: lags, rolling stats, calendar, price, promotions
- L1/L2 regularisation to prevent overfitting on short series
- Best for: capturing promotional effects and external regressors

### XGBoost Regressor
- Depth-limited trees with subsample + colsample regularisation
- Hyperparameter tuning via Optuna (Bayesian optimisation, 50 trials)
- SHAP explainability: understand *why* a forecast is high or low
- Best for: datasets with complex non-linear interaction effects

### Ensemble Strategy
Weighted average based on out-of-fold validation performance:
```
final_forecast = w_sarima * f_sarima + w_lgbm * f_lgbm + w_xgb * f_xgb

Weights recomputed each retraining cycle from validation MAPE.
```

---

## 📊 Performance Benchmarks

Results on 24-month retail dataset (weekly granularity, 50 SKUs):

| Model | MAPE | RMSE | MAE | Training Time |
|-------|------|------|-----|---------------|
| SARIMA | 8.2% | 1,240 | 890 | 45s |
| LightGBM | 5.8% | 920 | 680 | 12s |
| XGBoost | 6.1% | 960 | 710 | 18s |
| **Ensemble** | **4.9%** | **820** | **590** | — |
| Naïve baseline | 14.3% | 2,100 | 1,560 | — |

**Ensemble improves on best single model by ~16% MAPE reduction.**

---

## 🔄 Automated Retraining

Retraining triggers automatically when:
1. **Scheduled** — every Sunday at 2 AM (configurable)
2. **Drift detected** — PSI > 0.2 on any key feature
3. **Performance degradation** — live MAPE exceeds 10%

Each retraining run:
- Logs all parameters + metrics to MLflow
- Compares to production model — promotes only if better
- Sends Slack notification with performance delta

---

## 🚀 Quick Start

```bash
git clone https://github.com/abhimanyu343/sales-forecasting-mlops
cd sales-forecasting-mlops
pip install -r requirements.txt

# Generate sample data
python data/generate_data.py

# Train all models (logs to MLflow)
python training/train.py --config configs/train_config.yaml

# Start MLflow UI
mlflow ui --port 5000

# Serve predictions
uvicorn api.main:app --port 8001

# Get a forecast
curl -X POST http://localhost:8001/forecast \
  -H "Content-Type: application/json" \
  -d '{"sku_id": "SKU_001", "horizon_days": 30, "include_intervals": true}'

# Docker (full stack)
docker-compose up --build
```

---

## 📁 Project Structure

```
sales-forecasting-mlops/
├── configs/
│   └── train_config.yaml       # All training hyperparameters
├── data/
│   └── generate_data.py        # Synthetic sales data generator
├── feature_store/
│   ├── aggregator.py           # Transaction → time-series rollup
│   ├── features.py             # 40+ feature engineering functions
│   └── drift.py                # PSI + KS drift detection
├── models/
│   ├── base.py                 # Abstract base model class
│   ├── arima.py                # SARIMA with auto-order + intervals
│   ├── lightgbm_model.py       # LightGBM with Optuna tuning
│   ├── xgboost_model.py        # XGBoost with SHAP explainability
│   └── ensemble.py             # Weighted ensemble + calibration
├── training/
│   ├── train.py                # Main training orchestrator
│   ├── evaluate.py             # Cross-validation + metrics
│   └── registry.py             # MLflow model registry wrapper
├── api/
│   ├── main.py                 # FastAPI prediction service
│   ├── predictor.py            # Model loading + inference
│   └── schemas.py              # Pydantic types
├── scheduler/
│   └── retrain.py              # APScheduler automated retraining
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
