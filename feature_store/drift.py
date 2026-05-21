"""
Data drift detection for sales forecasting pipeline.

Detects when the statistical distribution of features or the target
has shifted enough to warrant model retraining.

Methods:
- Population Stability Index (PSI) — industry standard for feature drift
- Kolmogorov-Smirnov test — for continuous distribution comparison
- Chi-square test — for categorical/discrete features
- Target drift — monitors prediction error distribution

Retraining is recommended when:
- PSI > 0.2 on any key feature
- KS p-value < 0.05 on target distribution
- Rolling MAPE > threshold (performance drift)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

log = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Result of a full drift analysis run."""
    feature_psi: Dict[str, float] = field(default_factory=dict)
    ks_results: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # stat, p-value
    target_drift: bool = False
    target_ks_pvalue: float = 1.0
    features_drifted: List[str] = field(default_factory=list)
    overall_drift_detected: bool = False
    max_psi: float = 0.0
    recommendation: str = "no_action"

    def summary(self) -> str:
        lines = [
            f"Drift Analysis Report",
            f"  Overall drift detected: {self.overall_drift_detected}",
            f"  Recommendation: {self.recommendation}",
            f"  Max PSI: {self.max_psi:.4f}",
            f"  Features with PSI > 0.2: {self.features_drifted}",
            f"  Target KS p-value: {self.target_ks_pvalue:.4f}",
        ]
        return "\n".join(lines)


def compute_psi(reference: np.ndarray, current: np.ndarray,
                n_bins: int = 10, epsilon: float = 1e-6) -> float:
    """
    Population Stability Index (PSI).

    PSI measures how much the distribution of a variable has shifted.
    Interpretation:
      PSI < 0.1  → No significant change
      PSI < 0.2  → Slight change, monitor
      PSI >= 0.2 → Significant change, investigate / retrain

    Args:
        reference: Reference (training) distribution values
        current:   Current (production) distribution values
        n_bins:    Number of bins for discretisation
        epsilon:   Small value to prevent log(0)

    Returns:
        PSI score (non-negative float)
    """
    # Build bins from reference data
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)  # Remove duplicate edges
    if len(bins) < 3:
        return 0.0  # Not enough variation to compute PSI

    # Compute proportions
    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)

    ref_pct = (ref_counts / len(reference)) + epsilon
    cur_pct = (cur_counts / len(current)) + epsilon

    # Normalise so they sum to ~1
    ref_pct = ref_pct / ref_pct.sum()
    cur_pct = cur_pct / cur_pct.sum()

    psi = np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct))
    return float(round(psi, 6))


def ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test for distribution equality.

    Returns:
        (statistic, p_value) — p < 0.05 indicates significant drift
    """
    stat, p_value = stats.ks_2samp(reference, current)
    return float(round(stat, 4)), float(round(p_value, 4))


class DriftDetector:
    """
    Full drift detection suite for the forecasting pipeline.

    Compares a reference dataset (training window) against a current
    dataset (recent production window) across key features.
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_alpha: float = 0.05,
        key_features: Optional[List[str]] = None
    ):
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.key_features = key_features or [
            "lag_7d", "lag_28d", "roll_mean_28d", "roll_std_28d",
            "ewm_14d", "day_of_week", "month", "trend_7d"
        ]

    def analyse(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        target_col: str = "sales"
    ) -> DriftReport:
        """
        Run full drift analysis comparing reference vs current data.

        Args:
            reference_df: Training / baseline data
            current_df:   Recent production data
            target_col:   Target column name

        Returns:
            DriftReport with PSI scores, KS results, and recommendation
        """
        report = DriftReport()

        # ── Feature drift (PSI + KS) ──────────────────────────────────────────
        available = [f for f in self.key_features
                     if f in reference_df.columns and f in current_df.columns]

        for feat in available:
            ref_vals = reference_df[feat].dropna().values
            cur_vals = current_df[feat].dropna().values

            if len(ref_vals) < 20 or len(cur_vals) < 20:
                continue

            psi = compute_psi(ref_vals, cur_vals)
            ks_stat, ks_p = ks_test(ref_vals, cur_vals)

            report.feature_psi[feat] = psi
            report.ks_results[feat] = (ks_stat, ks_p)

            if psi >= self.psi_threshold:
                report.features_drifted.append(feat)
                log.warning(f"Drift detected in '{feat}': PSI={psi:.4f} (threshold={self.psi_threshold})")

        # ── Target drift ──────────────────────────────────────────────────────
        if target_col in reference_df.columns and target_col in current_df.columns:
            ref_target = reference_df[target_col].dropna().values
            cur_target = current_df[target_col].dropna().values
            _, target_ks_p = ks_test(ref_target, cur_target)
            report.target_ks_pvalue = target_ks_p
            report.target_drift = target_ks_p < self.ks_alpha
            if report.target_drift:
                log.warning(f"Target drift detected: KS p-value={target_ks_p:.4f}")

        # ── Overall assessment ────────────────────────────────────────────────
        report.max_psi = max(report.feature_psi.values()) if report.feature_psi else 0.0
        report.overall_drift_detected = (
            bool(report.features_drifted) or report.target_drift
        )

        if report.max_psi > 0.4 or report.target_drift:
            report.recommendation = "retrain_immediately"
        elif report.max_psi > 0.2 or len(report.features_drifted) > 2:
            report.recommendation = "schedule_retraining"
        elif report.max_psi > 0.1:
            report.recommendation = "monitor_closely"
        else:
            report.recommendation = "no_action"

        log.info(report.summary())
        return report

    def check_performance_drift(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        mape_threshold: float = 0.10,
        window: int = 30
    ) -> Dict:
        """
        Check if model performance has degraded by computing rolling MAPE.

        Args:
            actual:    Actual sales values
            predicted: Model predictions
            mape_threshold: Alert if MAPE exceeds this
            window:    Rolling window for MAPE computation

        Returns:
            Dict with rolling_mape, max_mape, degradation_detected
        """
        mape_series = np.abs((actual - predicted) / (np.abs(actual) + 1e-9))
        rolling_mape = pd.Series(mape_series).rolling(window, min_periods=window // 2).mean()

        current_mape = rolling_mape.iloc[-1] if len(rolling_mape) > 0 else 0.0
        max_mape = rolling_mape.max()

        degraded = current_mape > mape_threshold
        if degraded:
            log.warning(f"Performance drift: rolling MAPE={current_mape*100:.1f}% exceeds threshold={mape_threshold*100:.0f}%")

        return {
            "current_rolling_mape_pct": round(current_mape * 100, 2),
            "max_rolling_mape_pct": round(max_mape * 100, 2),
            "threshold_pct": mape_threshold * 100,
            "degradation_detected": degraded,
            "window_days": window
        }
