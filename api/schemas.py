"""Pydantic schemas for the Sales Forecast API."""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class HistoricalPoint(BaseModel):
    date: str = Field(..., description="ISO date string: YYYY-MM-DD")
    sales: float = Field(..., ge=0, description="Sales quantity (non-negative)")
    price: Optional[float] = Field(None, ge=0)
    promotion_flag: Optional[int] = Field(None, ge=0, le=1)


class ForecastRequest(BaseModel):
    sku_id: str = Field(..., min_length=1, max_length=100)
    history: List[HistoricalPoint] = Field(..., min_items=30,
        description="Historical sales data. Min 30 days, 90+ days recommended.")
    horizon_days: int = Field(30, ge=1, le=365,
        description="Number of days to forecast ahead")
    include_intervals: bool = Field(True, description="Include 80% prediction intervals")

    @validator("horizon_days")
    def reasonable_horizon(cls, v, values):
        history_len = len(values.get("history", []))
        if v > history_len:
            raise ValueError(f"Horizon ({v}) cannot exceed history length ({history_len})")
        return v


class ForecastPoint(BaseModel):
    date: str
    forecast: float
    lower_80: Optional[float] = None
    upper_80: Optional[float] = None


class ForecastResponse(BaseModel):
    sku_id: str
    horizon_days: int
    forecast: List[ForecastPoint]
    model_weights: Dict[str, float]
    latency_ms: float
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class BatchForecastRequest(BaseModel):
    skus: List[ForecastRequest] = Field(..., max_items=100)


class BatchForecastResponse(BaseModel):
    results: List[ForecastResponse]
    errors: List[Dict[str, str]]
    n_successful: int
    n_failed: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ModelInfoResponse(BaseModel):
    weights: Dict[str, float]
    metrics: Dict[str, Any]
    feature_count: int
    is_fitted: bool
    model_loaded_at: Optional[datetime]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_loaded_at: Optional[datetime]
    model_path: str
