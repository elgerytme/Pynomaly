import json
import os
import unittest
from unittest.mock import patch
from pynomaly.infrastructure.monitoring.prometheus_metrics import PrometheusMetricsService, initialize_metrics

class TestPrometheusMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics_service = initialize_metrics(enable_default_metrics=False)

    @patch('pynomaly.infrastructure.monitoring.prometheus_metrics.Counter.labels')
    def test_increment_http_requests_total(self, mock_labels):
        self.metrics_service.record_http_request(
            method="POST",
            endpoint="/api/test",
            status_code=200,
            duration=0.123
        )
        mock_labels.assert_called_with(method="POST", endpoint="/api/test", status="200")
        mock_labels.return_value.inc.assert_called_once()

    def test_generate_grafana_dashboard_json(self):
        from pynomaly.infrastructure.monitoring.dashboards import DashboardGenerator

        dashboard_configs = DashboardGenerator.generate_all_dashboards()
        for name, config in dashboard_configs.items():
            parsed_config = json.loads(config)
            assert "dashboard" in parsed_config
            assert "panels" in parsed_config["dashboard"]

if __name__ == '__main__':
    unittest.main()

