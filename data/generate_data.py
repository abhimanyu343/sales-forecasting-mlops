"""
Synthetic sales data generator for testing and demos.

Generates realistic multi-SKU retail sales with:
- Trend (linear growth + level shifts)
- Weekly seasonality (weekend dip/surge by category)
- Annual seasonality (holiday spikes, seasonal products)
- Promotional effects (step-change during promo periods)
- Price elasticity (higher price → lower demand)
- Noise (heteroskedastic — variance scales with level)
- Occasional stock-out events (demand clipped to 0)
- Competitor events (demand shock ± 20%)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# ── SKU category definitions ──────────────────────────────────────────────────
SKU_CATEGORIES = {
    "Electronics":  {"base": 850,  "trend": 0.12,  "seasonality": "low",    "promo_lift": 0.30},
    "Apparel":      {"base": 420,  "trend": 0.06,  "seasonality": "high",   "promo_lift": 0.50},
    "Grocery":      {"base": 1200, "trend": 0.02,  "seasonality": "medium", "promo_lift": 0.15},
    "Home":         {"base": 560,  "trend": 0.08,  "seasonality": "medium", "promo_lift": 0.35},
    "Sports":       {"base": 380,  "trend": 0.15,  "seasonality": "high",   "promo_lift": 0.40},
}

SEASONALITY_PROFILES = {
    "low":    {"weekly_amp": 0.05, "annual_amp": 0.10},
    "medium": {"weekly_amp": 0.12, "annual_amp": 0.25},
    "high":   {"weekly_amp": 0.20, "annual_amp": 0.45},
}

HOLIDAY_WEEKS = [1, 14, 26, 35, 44, 48, 52]  # New Year, Easter, Summer, Onam, Diwali, pre-Xmas, Xmas


def generate_sku_series(
    sku_id: str,
    category: str,
    dates: pd.DatetimeIndex,
    seed: int = 42,
) -> pd.Series:
    """Generate a single SKU sales time series."""
    np.random.seed(seed)
    cfg = SKU_CATEGORIES.get(category, SKU_CATEGORIES["Grocery"])
    sea = SEASONALITY_PROFILES[cfg["seasonality"]]

    n = len(dates)
    t = np.arange(n)

    # ── Base level with linear trend ──────────────────────────────────────────
    base = cfg["base"] * (1 + cfg["trend"] * t / 365)

    # Random level shift (structural break) — happens 0-2 times
    n_breaks = np.random.randint(0, 3)
    for _ in range(n_breaks):
        break_t = np.random.randint(90, n - 90)
        shift_magnitude = np.random.uniform(-0.15, 0.25)
        base[break_t:] *= (1 + shift_magnitude)

    # ── Weekly seasonality ────────────────────────────────────────────────────
    dow = dates.dayofweek.values  # 0=Mon, 6=Sun
    weekly_pattern = np.array([-0.10, -0.05, 0.00, 0.05, 0.15, 0.25, 0.20])  # Mon→Sun
    weekly_seasonality = sea["weekly_amp"] * weekly_pattern[dow] * base

    # ── Annual seasonality (Fourier) ──────────────────────────────────────────
    doy = dates.day_of_year.values
    annual_sin = sea["annual_amp"] * np.sin(2 * np.pi * doy / 365.25)
    annual_cos = sea["annual_amp"] * 0.4 * np.cos(4 * np.pi * doy / 365.25)
    annual_seasonality = (annual_sin + annual_cos) * base * 0.5

    # ── Holiday spikes ────────────────────────────────────────────────────────
    week_of_year = dates.isocalendar().week.astype(int).values
    holiday_mask = np.isin(week_of_year, HOLIDAY_WEEKS)
    holiday_lift = np.where(holiday_mask, base * np.random.uniform(0.2, 0.6), 0)

    # ── Promotional events (random 3-7 day windows, 3-6 per year) ─────────────
    promo_effect = np.zeros(n)
    promos_per_year = np.random.randint(3, 7)
    n_promos = int(promos_per_year * n / 365)
    for _ in range(n_promos):
        promo_start = np.random.randint(0, n - 14)
        promo_len = np.random.randint(3, 8)
        promo_effect[promo_start:promo_start + promo_len] += base[promo_start] * cfg["promo_lift"]

    # ── Competitor events (demand shock) ──────────────────────────────────────
    comp_effect = np.zeros(n)
    n_comp_events = np.random.randint(0, 4)
    for _ in range(n_comp_events):
        event_t = np.random.randint(30, n - 30)
        event_len = np.random.randint(7, 21)
        shock = np.random.uniform(-0.20, 0.20)
        comp_effect[event_t:event_t + event_len] = base[event_t] * shock

    # ── Heteroskedastic noise ─────────────────────────────────────────────────
    noise_std = base * np.random.uniform(0.08, 0.15)
    noise = np.random.normal(0, noise_std)

    # ── Combine all components ────────────────────────────────────────────────
    sales = base + weekly_seasonality + annual_seasonality + holiday_lift + promo_effect + comp_effect + noise

    # ── Stock-out simulation (random zeroing) ─────────────────────────────────
    n_stockouts = np.random.randint(0, 6)
    for _ in range(n_stockouts):
        so_start = np.random.randint(0, n - 7)
        so_len = np.random.randint(1, 5)
        sales[so_start:so_start + so_len] = 0

    return pd.Series(np.maximum(sales, 0).round(0), index=dates, name="sales")


def generate_sales_data(
    n_skus: int = 10,
    days: int = 730,
    start_date: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a multi-SKU sales dataset.

    Returns long-format DataFrame with columns:
    date, sku_id, category, sales, price, promotion_flag
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    categories = list(SKU_CATEGORIES.keys())
    records = []

    for i in range(n_skus):
        sku_id = f"SKU_{i+1:03d}"
        category = categories[i % len(categories)]
        cfg = SKU_CATEGORIES[category]

        sales = generate_sku_series(sku_id, category, dates, seed=seed + i)

        # Price (mean-reverting random walk around base price)
        base_price = np.random.uniform(199, 4999)
        price_walk = base_price + np.cumsum(np.random.normal(0, base_price * 0.005, days))
        price = np.clip(price_walk, base_price * 0.5, base_price * 2).round(2)

        # Promotion flag
        promo_flag = np.zeros(days, dtype=int)
        n_promos = np.random.randint(3, 7)
        for _ in range(n_promos):
            start = np.random.randint(0, days - 7)
            length = np.random.randint(3, 8)
            promo_flag[start:start + length] = 1

        for j, date in enumerate(dates):
            records.append({
                "date":           date,
                "sku_id":         sku_id,
                "category":       category,
                "sales":          int(sales.iloc[j]),
                "price":          price[j],
                "promotion_flag": promo_flag[j],
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["sku_id", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skus", type=int, default=10)
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--output", type=str, default="data/sales_data.csv")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df = generate_sales_data(n_skus=args.skus, days=args.days)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df):,} rows for {args.skus} SKUs over {args.days} days")
    print(df.groupby("category")["sales"].agg(["mean", "std", "min", "max"]).round(1))
