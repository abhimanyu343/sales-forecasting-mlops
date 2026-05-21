"""
FastAPI prediction service for the Sales Forecasting MLOps pipeline.

Endpoints:
  POST /forecast          Get sales forecast for a SKU
  POST /forecast/batch    Batch forecasts for multiple SKUs
  GET  /model/info        Model metadata, weights, performance metrics
  GET  /model/features    Feature importance from the ensemble
  POST /model/reload      Hot-reload model from disk without downtime
  GET  /health            Health check

The server loads the ensemble model at startup and serves predictions
with sub-50ms latency for single SKU requests.
"""

import os
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    ForecastRequest, ForecastResponse, BatchForecastRequest,
    BatchForecastResponse, ModelInfoResponse, HealthResponse
)

log = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/ensemble.pkl")

# Global model reference
_model = None
_model_loaded_at = None


def load_model(path: str = MODEL_PATH):
    """Load ensemble model from disk."""
    global _model, _model_loaded_at
    from models.ensemble import SalesEnsemble
    if not Path(path).exists():
        log.warning(f"Model not found at {path}. Run training/train.py first.")
        return None
    _model = SalesEnsemble.load(path)
    _model_loaded_at = datetime.utcnow()
    log.info(f"Model loaded from {path} | Weights: {_model.weights}")
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    log.info("Starting Sales Forecast API...")
    load_model()
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Sales Forecasting API",
    description="Production ML API for retail sales forecasting — SARIMA + LightGBM + XGBoost ensemble",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def timing_middleware(request, call_next):
    t = time.time()
    resp = await call_next(request)
    resp.headers["X-Response-Time-Ms"] = str(round((time.time() - t) * 1000, 1))
    return resp


def _require_model():
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first: python training/train.py"
        )
    return _model


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_loaded_at=_model_loaded_at,
        model_path=MODEL_PATH,
    )


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """Hot-reload model from disk without restarting the server."""
    m = load_model()
    if m is None:
        raise HTTPException(503, f"Model file not found at {MODEL_PATH}")
    return {"status": "reloaded", "weights": _model.weights, "loaded_at": _model_loaded_at}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    m = _require_model()
    return ModelInfoResponse(
        weights=m.weights,
        metrics=m.metrics,
        feature_count=len(m.feature_cols),
        is_fitted=m.is_fitted,
        model_loaded_at=_model_loaded_at,
    )


@app.get("/model/features", tags=["Model"])
async def feature_importance(top_k: int = Query(20, ge=5, le=50)):
    """Return top feature importances from the LightGBM sub-model."""
    m = _require_model()
    if "lightgbm" not in m.base_models:
        raise HTTPException(404, "LightGBM model not found in ensemble.")
    lgbm = m.base_models["lightgbm"]
    importances = lgbm.feature_importances_
    pairs = sorted(zip(m.feature_cols, importances), key=lambda x: x[1], reverse=True)
    return {
        "features": [{"name": f, "importance": round(float(i), 4)} for f, i in pairs[:top_k]],
        "total_features": len(m.feature_cols),
    }


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
async def forecast(request: ForecastRequest):
    """
    Generate a sales forecast for a single SKU.

    Provide recent historical data (at least 90 days recommended) and
    get back a day-by-day forecast with 80% prediction intervals.
    """
    m = _require_model()
    t0 = time.time()

    if not request.history:
        raise HTTPException(400, "No historical data provided.")
    if len(request.history) < 30:
        raise HTTPException(400, f"Need at least 30 days of history (got {len(request.history)}).")

    # Build DataFrame from request history
    hist_df = pd.DataFrame([
        {"date": pd.Timestamp(h.date), "sales": h.sales,
         "price": h.price or 0.0, "promotion_flag": h.promotion_flag or 0}
        for h in request.history
    ]).sort_values("date")

    # Feature engineering
    from feature_store.features import build_features, get_feature_columns
    feat_df = build_features(hist_df, target_col="sales", date_col="date",
                              include_fourier=True,
                              include_price="price" in hist_df.columns)
    feat_df = feat_df.dropna()

    forecast_df = m.predict(feat_df, horizon=request.horizon_days)
    latency_ms = round((time.time() - t0) * 1000, 1)

    return ForecastResponse(
        sku_id=request.sku_id,
        horizon_days=request.horizon_days,
        forecast=[
            {
                "date": str(row["date"].date()),
                "forecast": round(row["forecast"], 1),
                "lower_80": round(row["lower_80"], 1),
                "upper_80": round(row["upper_80"], 1),
            }
            for _, row in forecast_df.iterrows()
        ],
        model_weights=m.weights,
        latency_ms=latency_ms,
    )


@app.post("/forecast/batch", response_model=BatchForecastResponse, tags=["Forecasting"])
async def batch_forecast(request: BatchForecastRequest, background_tasks: BackgroundTasks):
    """Batch forecast for multiple SKUs. Returns aggregated and per-SKU results."""
    if len(request.skus) > 100:
        raise HTTPException(400, "Max 100 SKUs per batch request.")

    results = []
    errors = []
    for sku_req in request.skus:
        try:
            result = await forecast(sku_req)
            results.append(result)
        except HTTPException as e:
            errors.append({"sku_id": sku_req.sku_id, "error": e.detail})

    return BatchForecastResponse(
        results=results,
        errors=errors,
        n_successful=len(results),
        n_failed=len(errors),
    )
