#!/usr/bin/env python3
"""
Update Monitoring Dashboards Script
Updates Grafana dashboards after deployment
"""

import json
import logging
import os
import sys
from pathlib import Path

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GrafanaDashboardUpdater:
    """Updates Grafana dashboards for anomaly_detection"""

    def __init__(self):
        self.grafana_url = os.getenv("GRAFANA_URL", "http://grafana:3000")
        self.grafana_token = os.getenv("GRAFANA_API_TOKEN")
        self.dashboards_dir = (
            Path(__file__).parent.parent / "config" / "monitoring" / "dashboards"
        )

    def update_all_dashboards(self) -> bool:
        """Update all Grafana dashboards"""
        logger.info("Updating Grafana dashboards...")

        if not self.grafana_token:
            logger.warning("No Grafana API token provided, skipping dashboard updates")
            return True

        dashboard_files = list(self.dashboards_dir.glob("*.json"))

        if not dashboard_files:
            logger.info("No dashboard files found")
            return True

        success = True

        for dashboard_file in dashboard_files:
            try:
                logger.info(f"Updating dashboard: {dashboard_file.name}")
                result = self.update_dashboard(dashboard_file)
                if not result:
                    success = False
            except Exception as e:
                logger.error(f"Failed to update dashboard {dashboard_file.name}: {e}")
                success = False

        return success

    def update_dashboard(self, dashboard_file: Path) -> bool:
        """Update a single dashboard"""
        try:
            with open(dashboard_file) as f:
                dashboard_data = json.load(f)

            # Prepare dashboard for API
            payload = {
                "dashboard": dashboard_data,
                "overwrite": True,
                "message": f"Updated by deployment script at {dashboard_file.name}",
            }

            headers = {
                "Authorization": f"Bearer {self.grafana_token}",
                "Content-Type": "application/json",
            }

            response = requests.post(
                f"{self.grafana_url}/api/dashboards/db",
                json=payload,
                headers=headers,
                timeout=30,
            )

            if response.status_code == 200:
                logger.info(f"Successfully updated dashboard: {dashboard_file.name}")
                return True
            else:
                logger.error(
                    f"Failed to update dashboard {dashboard_file.name}: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"Error updating dashboard {dashboard_file.name}: {e}")
            return False


def main():
    """Main function"""
    try:
        updater = GrafanaDashboardUpdater()
        success = updater.update_all_dashboards()

        if success:
            logger.info("✅ All dashboards updated successfully")
            sys.exit(0)
        else:
            logger.error("❌ Some dashboards failed to update")
            sys.exit(1)

    except Exception as e:
        logger.error(f"💥 Dashboard update failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
