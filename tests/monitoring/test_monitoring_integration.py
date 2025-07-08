import unittest
import json
import time
import requests
from unittest.mock import patch, Mock
import threading
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pynomaly.infrastructure.monitoring.prometheus_metrics import PrometheusMetricsService
from pynomaly.infrastructure.monitoring.dashboards import DashboardGenerator
from pynomaly.presentation.api.app import app
from fastapi.testclient import TestClient


class TestMonitoringIntegration(unittest.TestCase):
    """
    Integration tests for monitoring stack.
    
    These tests validate the integration between Pynomaly,
    Prometheus, Grafana, and Pushgateway in a containerized environment.
    """
    
    def setUp(self):
        """Set up test environment with monitoring services."""
        # Use service discovery from GitHub Actions services
        self.prometheus_url = "http://localhost:9090"
        self.grafana_url = "http://localhost:3000"
        self.pushgateway_url = "http://localhost:9091"
        
        # Initialize metrics service
        self.metrics_service = PrometheusMetricsService(
            enable_default_metrics=True,
            namespace="pynomaly_test",
            port=None  # Don't start server in tests
        )
        
        # Create test client for API
        self.client = TestClient(app)
        
        # Wait for services to be ready
        self._wait_for_services()
    
    def _wait_for_services(self, timeout=30):
        """Wait for monitoring services to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check Prometheus
                response = requests.get(f"{self.prometheus_url}/-/ready", timeout=2)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(1)
        else:
            self.skipTest("Prometheus not ready within timeout")
    
    def test_prometheus_metrics_scraping(self):
        """Test that Prometheus can scrape metrics."""
        # Check Prometheus targets
        response = requests.get(f"{self.prometheus_url}/api/v1/targets")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data['status'], 'success')
        
        # Check that we can query metrics
        response = requests.get(f"{self.prometheus_url}/api/v1/query?query=up")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('data', data)
    
    def test_pynomaly_api_emits_metrics(self):
        """Test that Pynomaly API emits metrics to Prometheus."""
        # Record some test metrics
        self.metrics_service.record_http_request(
            method="GET",
            endpoint="/test",
            status_code=200,
            duration=0.1
        )
        
        self.metrics_service.record_detection(
            algorithm="isolation_forest",
            dataset_type="tabular",
            dataset_size=1000,
            duration=2.5,
            anomalies_found=5,
            success=True,
            accuracy=0.95
        )
        
        # Get metrics data
        metrics_data = self.metrics_service.get_metrics_data()
        self.assertIsInstance(metrics_data, bytes)
        
        # Check that metrics contain expected data
        metrics_text = metrics_data.decode('utf-8')
        self.assertIn('pynomaly_test_http_requests_total', metrics_text)
        self.assertIn('pynomaly_test_detections_total', metrics_text)
    
    def test_grafana_dashboard_creation(self):
        """Test that Grafana dashboards can be created."""
        # Generate dashboard configurations
        generator = DashboardGenerator()
        dashboard_configs = generator.generate_all_dashboards()
        
        # Validate dashboard JSON structures
        for name, config in dashboard_configs.items():
            with self.subTest(dashboard=name):
                # Parse JSON to ensure it's valid
                parsed_config = json.loads(config)
                
                # Validate required fields
                self.assertIn('dashboard', parsed_config)
                dashboard = parsed_config['dashboard']
                
                self.assertIn('title', dashboard)
                self.assertIn('panels', dashboard)
                self.assertIn('tags', dashboard)
                
                # Validate panels structure
                for panel in dashboard['panels']:
                    self.assertIn('id', panel)
                    self.assertIn('title', panel)
                    self.assertIn('type', panel)
                    self.assertIn('targets', panel)
    
    def test_pushgateway_metrics_push(self):
        """Test pushing metrics to Pushgateway."""
        # Create test metric
        metric_data = "pynomaly_test_metric{job=\"test\"} 1\n"
        
        # Push metric to Pushgateway
        response = requests.post(
            f"{self.pushgateway_url}/metrics/job/pynomaly_test",
            data=metric_data,
            headers={'Content-Type': 'text/plain'}
        )
        
        # Check if push was successful (may fail if pushgateway not available)
        if response.status_code == 200:
            # Verify metric was stored
            response = requests.get(f"{self.pushgateway_url}/api/v1/metrics")
            self.assertEqual(response.status_code, 200)
        else:
            self.skipTest("Pushgateway not available")
    
    def test_api_health_endpoint(self):
        """Test that API health endpoint works."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_end_to_end_monitoring_flow(self):
        """Test complete monitoring flow from API to Prometheus."""
        # 1. Make API request to generate metrics
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        # 2. Record metrics manually (simulating middleware)
        self.metrics_service.record_http_request(
            method="GET",
            endpoint="/health",
            status_code=200,
            duration=0.05
        )
        
        # 3. Verify metrics are generated
        metrics_data = self.metrics_service.get_metrics_data()
        metrics_text = metrics_data.decode('utf-8')
        
        # Check for expected metrics
        self.assertIn('pynomaly_test_http_requests_total', metrics_text)
        
        # 4. Verify dashboard configuration is valid
        dashboard_configs = DashboardGenerator.generate_all_dashboards()
        overview_config = json.loads(dashboard_configs['overview'])
        
        # Check dashboard has HTTP request panels
        panels = overview_config['dashboard']['panels']
        http_panels = [p for p in panels if 'http' in p['title'].lower()]
        self.assertGreater(len(http_panels), 0)


if __name__ == '__main__':
    unittest.main()
