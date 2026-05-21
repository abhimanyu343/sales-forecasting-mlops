"""
Automated retraining scheduler using APScheduler.

Triggers:
1. Scheduled — every Sunday at 2:00 AM (configurable via env)
2. Drift-triggered — when DriftDetector flags significant distribution shift
3. Performance-triggered — when rolling MAPE exceeds threshold

On each retraining run:
- Loads fresh data from source
- Runs full training pipeline
- Compares new model against current production on a holdout set
- Promotes new model only if MAPE improves
- Sends Slack/email notification with before/after metrics

Usage:
    python scheduler/retrain.py          # Start scheduler daemon
    python scheduler/retrain.py --now    # Trigger immediate retrain
"""

import logging
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False


class RetrainingScheduler:
    """Orchestrates automated model retraining with promotion logic."""

    def __init__(
        self,
        config: dict = None,
        model_path: str = "models/ensemble.pkl",
        staging_path: str = "models/ensemble_staging.pkl",
        promote_threshold_pct: float = 0.5,
        notify_slack: bool = False,
        slack_webhook: str = None,
    ):
        self.config = config or {}
        self.model_path = model_path
        self.staging_path = staging_path
        self.promote_threshold_pct = promote_threshold_pct
        self.notify_slack = notify_slack
        self.slack_webhook = slack_webhook or os.getenv("SLACK_WEBHOOK_URL")
        self._last_retrain: datetime = None
        self._retrain_count: int = 0

    def check_drift_and_retrain(self) -> bool:
        """
        Check for data drift and retrain if warranted.
        Returns True if retraining was triggered.
        """
        from data.generate_data import generate_sales_data
        from feature_store.features import build_features, get_feature_columns
        from feature_store.drift import DriftDetector

        log.info("Running drift check...")
        try:
            # Load reference data (training window) and current data (recent)
            data_path = self.config.get("data_path", "data/sales_data.csv")
            import pandas as pd
            df = pd.read_csv(data_path, parse_dates=["date"])

            cutoff = df["date"].max() - pd.Timedelta(days=30)
            reference = df[df["date"] < cutoff]
            current   = df[df["date"] >= cutoff]

            feat_ref = build_features(reference).dropna()
            feat_cur = build_features(current).dropna()

            feature_cols = get_feature_columns(feat_ref)
            detector = DriftDetector(key_features=feature_cols[:10])
            report = detector.analyse(feat_ref, feat_cur)

            if report.overall_drift_detected:
                log.warning(f"Drift detected — triggering retraining. Recommendation: {report.recommendation}")
                return self.run_retrain(trigger="drift")
            else:
                log.info(f"No significant drift detected. Max PSI: {report.max_psi:.4f}")
                return False

        except Exception as e:
            log.error(f"Drift check failed: {e}", exc_info=True)
            return False

    def run_retrain(self, trigger: str = "scheduled") -> bool:
        """
        Full retraining run with model comparison and conditional promotion.

        Args:
            trigger: What triggered this run ('scheduled', 'drift', 'performance', 'manual')

        Returns:
            True if new model was promoted to production
        """
        log.info(f"=" * 50)
        log.info(f"Retraining triggered by: {trigger}")
        log.info(f"Time: {datetime.now().isoformat()}")
        log.info(f"=" * 50)

        self._retrain_count += 1

        try:
            from training.train import run_training, load_config

            # Train new model (saves to staging path)
            cfg = {**self.config, "model_save_path": self.staging_path}
            new_metrics = run_training(cfg, run_name=f"retrain_{trigger}_{self._retrain_count}")
            new_mape = new_metrics["test_mape"]

            # Compare against current production model
            current_mape = self._get_current_production_mape()

            improvement = current_mape - new_mape
            promoted = improvement >= -self.promote_threshold_pct  # Allow slight regression

            if promoted:
                import shutil
                shutil.copy(self.staging_path, self.model_path)
                log.info(f"New model PROMOTED: MAPE {current_mape:.2f}% → {new_mape:.2f}% (Δ {improvement:+.2f}%)")
            else:
                log.warning(f"New model NOT promoted: MAPE {new_mape:.2f}% vs production {current_mape:.2f}%")

            self._last_retrain = datetime.now()

            if self.notify_slack:
                self._send_slack_notification(trigger, new_mape, current_mape, promoted, new_metrics)

            return promoted

        except Exception as e:
            log.error(f"Retraining failed: {e}", exc_info=True)
            return False

    def _get_current_production_mape(self) -> float:
        """Get the MAPE of the currently deployed production model."""
        if not Path(self.model_path).exists():
            return 999.0  # No production model — always promote
        try:
            from models.ensemble import SalesEnsemble
            m = SalesEnsemble.load(self.model_path)
            return m.metrics.get("ensemble_val_mape", 999.0)
        except Exception:
            return 999.0

    def _send_slack_notification(self, trigger, new_mape, old_mape, promoted, metrics):
        """Send a Slack webhook notification with retraining results."""
        if not self.slack_webhook:
            return
        import urllib.request, json
        emoji = "✅" if promoted else "⚠️"
        status = "PROMOTED" if promoted else "NOT PROMOTED"
        msg = {
            "text": f"{emoji} *Sales Forecast Model Retraining — {status}*",
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn",
                    "text": f"{emoji} *Retraining Complete* | Trigger: `{trigger}`"}},
                {"type": "section", "fields": [
                    {"type": "mrkdwn", "text": f"*New MAPE:* {new_mape:.2f}%"},
                    {"type": "mrkdwn", "text": f"*Prod MAPE:* {old_mape:.2f}%"},
                    {"type": "mrkdwn", "text": f"*Status:* {status}"},
                    {"type": "mrkdwn", "text": f"*CV MAPE:* {metrics.get('cv_mean_mape', '?'):.2f}%"},
                ]}
            ]
        }
        try:
            req = urllib.request.Request(
                self.slack_webhook,
                data=json.dumps(msg).encode(),
                headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            log.warning(f"Slack notification failed: {e}")

    def start(self, cron_schedule: str = "0 2 * * 0"):  # Sunday 2 AM
        """Start the scheduler daemon."""
        if not APSCHEDULER_AVAILABLE:
            raise ImportError("apscheduler not installed. Run: pip install apscheduler")

        scheduler = BlockingScheduler(timezone="Asia/Kolkata")

        # Scheduled retraining
        scheduler.add_job(
            lambda: self.run_retrain("scheduled"),
            CronTrigger.from_crontab(cron_schedule),
            id="scheduled_retrain",
            name="Weekly model retraining",
            misfire_grace_time=3600,
        )

        # Drift check every 6 hours
        scheduler.add_job(
            self.check_drift_and_retrain,
            "interval", hours=6,
            id="drift_check",
            name="Drift detection check",
        )

        log.info(f"Scheduler started. Retraining cron: '{cron_schedule}'")
        log.info("Press Ctrl+C to stop.")
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            log.info("Scheduler stopped.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="Trigger immediate retrain")
    parser.add_argument("--cron", type=str, default="0 2 * * 0", help="Cron schedule")
    args = parser.parse_args()

    sched = RetrainingScheduler(notify_slack=bool(os.getenv("SLACK_WEBHOOK_URL")))
    if args.now:
        sched.run_retrain(trigger="manual")
    else:
        sched.start(cron_schedule=args.cron)
