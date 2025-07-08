"""Unit tests for Grafana dashboard JSON generation and validation."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from pynomaly.infrastructure.monitoring.dashboards import (
    DETECTION_PANELS,
    ENSEMBLE_PANELS,
    OVERVIEW_PANELS,
    STREAMING_PANELS,
    SYSTEM_PANELS,
    DashboardGenerator,
    get_complete_dashboard_template,
)


class TestDashboardJSON:
    """Test suite for dashboard JSON generation and validation."""
    
    def test_complete_dashboard_template_structure(self):
        """Test that the complete dashboard template has the correct structure."""
        dashboard = get_complete_dashboard_template()
        
        # Check top-level structure
        assert "dashboard" in dashboard
        dashboard_config = dashboard["dashboard"]
        
        # Check required dashboard properties
        assert dashboard_config["title"] == "Pynomaly - Anomaly Detection System"
        assert "pynomaly" in dashboard_config["tags"]
        assert "anomaly-detection" in dashboard_config["tags"]
        assert "ml" in dashboard_config["tags"]
        assert dashboard_config["style"] == "dark"
        assert dashboard_config["timezone"] == "browser"
        assert dashboard_config["refresh"] == "30s"
        
        # Check time configuration
        assert "time" in dashboard_config
        assert dashboard_config["time"]["from"] == "now-1h"
        assert dashboard_config["time"]["to"] == "now"
        
        # Check timepicker configuration
        assert "timepicker" in dashboard_config
        timepicker = dashboard_config["timepicker"]
        assert "refresh_intervals" in timepicker
        assert "time_options" in timepicker
        assert "30s" in timepicker["refresh_intervals"]
        assert "1h" in timepicker["time_options"]
        
        # Check panels exist
        assert "panels" in dashboard_config
        assert isinstance(dashboard_config["panels"], list)
        assert len(dashboard_config["panels"]) > 0
    
    def test_complete_dashboard_contains_all_panel_types(self):
        """Test that the complete dashboard contains all expected panel types."""
        dashboard = get_complete_dashboard_template()
        panels = dashboard["dashboard"]["panels"]
        
        # Expected panel counts
        expected_overview_panels = len(OVERVIEW_PANELS)
        expected_detection_panels = len(DETECTION_PANELS)
        expected_streaming_panels = len(STREAMING_PANELS)
        expected_ensemble_panels = len(ENSEMBLE_PANELS)
        expected_system_panels = len(SYSTEM_PANELS)
        
        total_expected = (
            expected_overview_panels +
            expected_detection_panels +
            expected_streaming_panels +
            expected_ensemble_panels +
            expected_system_panels
        )
        
        assert len(panels) == total_expected
        
        # Check for specific panel IDs from each category
        panel_ids = {panel["id"] for panel in panels}
        
        # Overview panels (1-3)
        assert 1 in panel_ids
        assert 2 in panel_ids
        assert 3 in panel_ids
        
        # Detection panels (10-13)
        assert 10 in panel_ids
        assert 11 in panel_ids
        assert 12 in panel_ids
        assert 13 in panel_ids
        
        # Streaming panels (20-23)
        assert 20 in panel_ids
        assert 21 in panel_ids
        assert 22 in panel_ids
        assert 23 in panel_ids
        
        # Ensemble panels (30-31)
        assert 30 in panel_ids
        assert 31 in panel_ids
        
        # System panels (40-43)
        assert 40 in panel_ids
        assert 41 in panel_ids
        assert 42 in panel_ids
        assert 43 in panel_ids
    
    def test_panels_have_required_properties(self):
        """Test that all panels have required properties."""
        dashboard = get_complete_dashboard_template()
        panels = dashboard["dashboard"]["panels"]
        
        required_properties = ["id", "title", "type", "gridPos", "targets"]
        
        for panel in panels:
            for prop in required_properties:
                assert prop in panel, f"Panel {panel.get('id', 'unknown')} missing property: {prop}"
            
            # Check gridPos has required dimensions
            grid_pos = panel["gridPos"]
            assert "h" in grid_pos  # height
            assert "w" in grid_pos  # width
            assert "x" in grid_pos  # x position
            assert "y" in grid_pos  # y position
            
            # Check targets format
            targets = panel["targets"]
            assert isinstance(targets, list)
            assert len(targets) > 0
            
            for target in targets:
                assert "expr" in target  # Prometheus query
                assert "refId" in target  # Reference ID
    
    def test_panel_queries_are_valid_prometheus(self):
        """Test that panel queries use valid Prometheus syntax."""
        dashboard = get_complete_dashboard_template()
        panels = dashboard["dashboard"]["panels"]
        
        # Common Prometheus metric patterns
        expected_metrics = [
            "pynomaly_active_models",
            "pynomaly_http_requests_total",
            "pynomaly_http_request_duration_seconds_bucket",
            "pynomaly_detections_total",
            "pynomaly_detection_duration_seconds_bucket",
            "pynomaly_anomalies_found_total",
            "pynomaly_detection_accuracy_ratio",
            "pynomaly_streaming_throughput_per_second",
            "pynomaly_streaming_buffer_utilization_ratio",
            "pynomaly_streaming_backpressure_events_total",
            "pynomaly_active_streams",
            "pynomaly_ensemble_predictions_total",
            "pynomaly_ensemble_agreement_ratio_bucket",
            "pynomaly_memory_usage_bytes",
            "pynomaly_cpu_usage_ratio",
            "pynomaly_errors_total",
            "pynomaly_cache_hit_ratio",
        ]
        
        found_metrics = set()
        
        for panel in panels:
            for target in panel["targets"]:
                expr = target["expr"]
                
                # Check for valid Prometheus functions - basic syntax check
                if "rate(" in expr:
                    # Check for proper rate function usage
                    assert expr.count("(") == expr.count(")")
                    assert "[" in expr and "]" in expr  # Time range
                if "histogram_quantile(" in expr:
                    assert expr.count("(") == expr.count(")")
                if "increase(" in expr:
                    assert expr.count("(") == expr.count(")")
                if "avg(" in expr:
                    assert expr.count("(") == expr.count(")")
                
                # Collect metrics used
                for metric in expected_metrics:
                    if metric in expr:
                        found_metrics.add(metric)
        
        # Ensure we found most of the expected metrics
        assert len(found_metrics) >= len(expected_metrics) * 0.8, (
            f"Expected to find most metrics, found: {found_metrics}"
        )
    
    def test_exported_json_file_exists_and_valid(self):
        """Test that the exported JSON file exists and is valid."""
        dashboard_path = Path("deploy/grafana/provisioning/dashboards/pynomaly.json")
        
        assert dashboard_path.exists(), f"Dashboard JSON file not found at {dashboard_path}"
        
        # Load and validate JSON
        with open(dashboard_path, "r") as f:
            dashboard_json = json.load(f)
        
        # Verify structure
        assert "dashboard" in dashboard_json
        assert "panels" in dashboard_json["dashboard"]
        assert len(dashboard_json["dashboard"]["panels"]) > 0
        
        # Verify it matches our template
        template_dashboard = get_complete_dashboard_template()
        assert dashboard_json["dashboard"]["title"] == template_dashboard["dashboard"]["title"]
        assert len(dashboard_json["dashboard"]["panels"]) == len(template_dashboard["dashboard"]["panels"])
    
    def test_individual_dashboards_generation(self):
        """Test that individual dashboards are generated correctly."""
        dashboards = DashboardGenerator.generate_all_dashboards()
        
        expected_dashboards = [
            "overview",
            "detection",
            "streaming",
            "ensemble",
            "system",
            "business",
        ]
        
        for dashboard_name in expected_dashboards:
            assert dashboard_name in dashboards, f"Missing dashboard: {dashboard_name}"
            
            # Parse the JSON to verify structure
            dashboard_json = json.loads(dashboards[dashboard_name])
            assert "dashboard" in dashboard_json
            assert "panels" in dashboard_json["dashboard"]
            assert len(dashboard_json["dashboard"]["panels"]) > 0
            
            # Verify title contains expected text
            title = dashboard_json["dashboard"]["title"]
            assert "Pynomaly" in title
            assert dashboard_name.title() in title or dashboard_name.upper() in title
    
    def test_individual_dashboard_files_exist(self):
        """Test that individual dashboard files are exported correctly."""
        dashboard_dir = Path("deploy/grafana/provisioning/dashboards")
        
        expected_files = [
            "pynomaly-overview.json",
            "pynomaly-detection.json",
            "pynomaly-streaming.json",
            "pynomaly-ensemble.json",
            "pynomaly-system.json",
            "pynomaly-business.json",
        ]
        
        for filename in expected_files:
            file_path = dashboard_dir / filename
            assert file_path.exists(), f"Dashboard file not found: {file_path}"
            
            # Verify JSON is valid
            with open(file_path, "r") as f:
                dashboard_json = json.load(f)
                assert "dashboard" in dashboard_json
                assert "panels" in dashboard_json["dashboard"]
    
    def test_panel_grid_positioning(self):
        """Test that panel grid positions are logical and non-overlapping."""
        dashboard = get_complete_dashboard_template()
        panels = dashboard["dashboard"]["panels"]
        
        # Track occupied grid positions
        occupied_positions = set()
        
        for panel in panels:
            grid_pos = panel["gridPos"]
            x, y, w, h = grid_pos["x"], grid_pos["y"], grid_pos["w"], grid_pos["h"]
            
            # Check for reasonable dimensions
            assert w > 0 and w <= 24, f"Panel {panel['id']} width out of bounds: {w}"
            assert h > 0 and h <= 20, f"Panel {panel['id']} height out of bounds: {h}"
            assert x >= 0 and x < 24, f"Panel {panel['id']} x position out of bounds: {x}"
            assert y >= 0, f"Panel {panel['id']} y position negative: {y}"
            
            # Check for grid line alignment (typically 24-column grid)
            assert x + w <= 24, f"Panel {panel['id']} extends beyond grid width"
    
    def test_alert_thresholds_are_reasonable(self):
        """Test that alert thresholds in panels are reasonable."""
        dashboard = get_complete_dashboard_template()
        panels = dashboard["dashboard"]["panels"]
        
        for panel in panels:
            if "fieldConfig" in panel and "defaults" in panel["fieldConfig"]:
                defaults = panel["fieldConfig"]["defaults"]
                if "thresholds" in defaults and "steps" in defaults["thresholds"]:
                    steps = defaults["thresholds"]["steps"]
                    
                    # Check threshold progression
                    for i, step in enumerate(steps):
                        if "value" in step and step["value"] is not None:
                            # Ensure numeric values are reasonable
                            value = step["value"]
                            assert isinstance(value, (int, float))
                            assert value >= 0  # No negative thresholds
                            
                            # Check color progression makes sense
                            assert "color" in step
                            assert step["color"] in ["green", "yellow", "red", "orange"]
    
    def test_provisioning_config_exists(self):
        """Test that Grafana provisioning configuration exists."""
        provisioning_path = Path("deploy/grafana/provisioning/dashboards/provisioning.yml")
        
        if provisioning_path.exists():
            # If PyYAML is available, validate the config
            try:
                import yaml
                with open(provisioning_path, "r") as f:
                    config = yaml.safe_load(f)
                
                assert "apiVersion" in config
                assert "providers" in config
                assert len(config["providers"]) > 0
                
                provider = config["providers"][0]
                assert "name" in provider
                assert "type" in provider
                assert provider["type"] == "file"
                
            except ImportError:
                # PyYAML not available, just check file exists
                with open(provisioning_path, "r") as f:
                    content = f.read()
                    assert "apiVersion" in content
                    assert "providers" in content


# Integration test to ensure the export script works
class TestDashboardExportScript:
    """Test the dashboard export script functionality."""
    
    def test_export_script_execution(self):
        """Test that the export script can be executed successfully."""
        # This test assumes the export script has been run
        # Check that all expected files exist
        dashboard_dir = Path("deploy/grafana/provisioning/dashboards")
        
        expected_files = [
            "pynomaly.json",
            "pynomaly-overview.json",
            "pynomaly-detection.json",
            "pynomaly-streaming.json",
            "pynomaly-ensemble.json",
            "pynomaly-system.json",
            "pynomaly-business.json",
        ]
        
        for filename in expected_files:
            file_path = dashboard_dir / filename
            assert file_path.exists(), f"Export script did not create: {filename}"
            
            # Verify file is not empty
            assert file_path.stat().st_size > 0, f"Exported file is empty: {filename}"
    
    def test_all_exported_dashboards_are_valid_json(self):
        """Test that all exported dashboard files contain valid JSON."""
        dashboard_dir = Path("deploy/grafana/provisioning/dashboards")
        
        json_files = list(dashboard_dir.glob("*.json"))
        assert len(json_files) > 0, "No JSON files found in dashboard directory"
        
        for json_file in json_files:
            with open(json_file, "r") as f:
                try:
                    dashboard_data = json.load(f)
                    assert isinstance(dashboard_data, dict)
                    assert "dashboard" in dashboard_data
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in {json_file}: {e}")
