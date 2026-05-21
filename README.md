# Sales Forecasting MLOps Pipeline

> End-to-end ML pipeline — ARIMA + XGBoost ensemble forecasting with MLflow experiment tracking, automated retraining, and Power BI integration.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![MLflow](https://img.shields.io/badge/MLflow-2.13-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green) ![Docker](https://img.shields.io/badge/Docker-ready-blue)

## Overview

Built on real-world demand forecasting work at GPIL where an ARIMA + regression model reduced material waste and aligned production schedules to actual consumption patterns. This repo productionises that approach with full MLOps tooling.

## Pipeline Architecture

```
Raw Sales Data (CSV / DB)
        │
        ▼
   Feature Engineering
   (lag features, rolling stats, calendar features)
        │
        ▼
  ┌─────┴──────┐
  │            │
ARIMA      XGBoost
  │            │
  └─────┬──────┘
        │  Ensemble (weighted average)
        ▼
   MLflow Tracking
   (params, metrics, artifacts)
        │
        ▼
  Model Registry → Staging → Production
        │
        ▼
  Scheduled Retraining (APScheduler)
        │
        ▼
  Power BI / Streamlit Dashboard
```

## Key Features

- **Ensemble model** — ARIMA handles trend/seasonality, XGBoost captures non-linear patterns
- **MLflow tracking** — every experiment logged with params, RMSE, MAPE, MAE
- **Model Registry** — automatic promotion to production when MAPE < threshold
- **Auto-retraining** — weekly scheduled retraining with data drift detection
- **Power BI connector** — exports forecast to .pbix-ready CSV + REST endpoint
- **Dockerised** — single `docker-compose up` to run everything

## Performance (on test set)

| Model | RMSE | MAPE | MAE |
|-------|------|------|-----|
| ARIMA | 12,340 | 8.2% | 9,870 |
| XGBoost | 9,120 | 6.1% | 7,450 |
| **Ensemble** | **7,890** | **5.3%** | **6,200** |

## Quick Start

```bash
git clone https://github.com/abhimanyu343/sales-forecasting-mlops
cd sales-forecasting-mlops

# With Docker
docker-compose up --build

# Without Docker
pip install -r requirements.txt
python pipeline/run.py --train --data data/sample_sales.csv
mlflow ui  # View experiments at localhost:5000
```

---
*[LinkedIn](https://linkedin.com/in/abhimanyusarda343) · Built from production experience at GPIL*
