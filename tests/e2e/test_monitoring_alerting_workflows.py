"""Monitoring and alerting workflow end-to-end tests.

This module tests real-time monitoring, alerting systems, and production
monitoring workflows for anomaly detection systems.
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app


class TestMonitoringAlertingWorkflows:
    """Test monitoring and alerting system workflows."""

    @pytest.fixture
    def app_client(self):
        """Create test client for API."""
        container = create_container()
        app = create_app(container)
        return TestClient(app)

    @pytest.fixture
    def streaming_data_generator(self):
        """Generate streaming data for monitoring tests."""

        def generate_batch(batch_size=100, anomaly_rate=0.05):
            np.random.seed(int(time.time()) % 1000)  # Semi-random seed

            # Generate normal data
            normal_size = int(batch_size * (1 - anomaly_rate))
            normal_data = np.random.multivariate_normal(
                [0, 0], [[1, 0.3], [0.3, 1]], normal_size
            )

            # Generate anomalous data
            anomaly_size = batch_size - normal_size
            anomaly_data = np.random.multivariate_normal(
                [3, 3], [[0.5, 0], [0, 0.5]], anomaly_size
            )

            # Combine and add metadata
            all_data = np.vstack([normal_data, anomaly_data])
            timestamps = [time.time() + i * 0.1 for i in range(batch_size)]

            df = pd.DataFrame(
                {
                    "feature_1": all_data[:, 0],
                    "feature_2": all_data[:, 1],
                    "timestamp": timestamps,
                    "batch_id": [
                        f"batch_{int(time.time())}" for _ in range(batch_size)
                    ],
                }
            )

            return df

        return generate_batch

    def test_real_time_monitoring_setup(self, app_client, streaming_data_generator):
        """Test setting up real-time monitoring system."""
        # Create baseline dataset for training
        baseline_data = streaming_data_generator(1000, 0.02)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            baseline_data.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            # Upload baseline dataset
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("baseline_data.csv", file, "text/csv")},
                    data={"name": "Monitoring Baseline Dataset"},
                )
            assert upload_response.status_code == 200
            baseline_dataset_id = upload_response.json()["id"]

            # Create monitoring detector
            detector_data = {
                "name": "Production Monitor",
                "algorithm_name": "IsolationForest",
                "parameters": {
                    "contamination": 0.05,
                    "random_state": 42,
                    "n_estimators": 100,
                },
            }

            create_response = app_client.post("/api/detectors/", json=detector_data)
            assert create_response.status_code == 200
            detector_id = create_response.json()["id"]

            # Train monitoring detector
            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train",
                json={"dataset_id": baseline_dataset_id},
            )
            assert train_response.status_code == 200

            # Set up monitoring configuration
            monitoring_config = {
                "detector_id": detector_id,
                "monitoring_name": "Production Anomaly Monitor",
                "data_source": {
                    "type": "streaming",
                    "batch_size": 100,
                    "interval_seconds": 60,
                },
                "thresholds": {
                    "anomaly_rate_threshold": 0.10,
                    "score_threshold": 0.75,
                    "consecutive_anomalies": 5,
                },
                "alert_channels": ["email", "webhook", "log"],
                "metrics": {
                    "track_drift": True,
                    "track_performance": True,
                    "track_volume": True,
                },
            }

            monitor_setup_response = app_client.post(
                "/api/monitoring/setup", json=monitoring_config
            )
            assert monitor_setup_response.status_code == 200
            monitor_result = monitor_setup_response.json()

            # Verify monitoring setup
            assert "monitor_id" in monitor_result
            assert "status" in monitor_result
            assert "configuration" in monitor_result
            assert monitor_result["status"] == "active"

            monitor_id = monitor_result["monitor_id"]

            # Get monitoring status
            status_response = app_client.get(f"/api/monitoring/{monitor_id}/status")
            assert status_response.status_code == 200
            status_result = status_response.json()

            assert "monitor_id" in status_result
            assert "status" in status_result
            assert "metrics" in status_result
            assert "last_check" in status_result

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_streaming_anomaly_detection(self, app_client, streaming_data_generator):
        """Test streaming anomaly detection with real-time processing."""
        # Set up detector (reuse from previous test pattern)
        baseline_data = streaming_data_generator(500, 0.02)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            baseline_data.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            # Upload and train detector
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("streaming_baseline.csv", file, "text/csv")},
                    data={"name": "Streaming Baseline"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            detector_data = {
                "name": "Streaming Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.05},
            }

            create_response = app_client.post("/api/detectors/", json=detector_data)
            assert create_response.status_code == 200
            detector_id = create_response.json()["id"]

            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train", json={"dataset_id": dataset_id}
            )
            assert train_response.status_code == 200

            # Start streaming session
            streaming_config = {
                "detector_id": detector_id,
                "session_name": "test_streaming_session",
                "buffer_size": 1000,
                "processing_mode": "real_time",
                "output_format": "json",
            }

            session_response = app_client.post(
                "/api/streaming/start", json=streaming_config
            )
            assert session_response.status_code == 200
            session_result = session_response.json()

            assert "session_id" in session_result
            assert "status" in session_result
            session_id = session_result["session_id"]

            # Send multiple batches of streaming data
            batch_results = []
            for i in range(5):
                # Generate batch with increasing anomaly rate
                anomaly_rate = 0.02 + (i * 0.02)  # 2%, 4%, 6%, 8%, 10%
                batch_data = streaming_data_generator(50, anomaly_rate)

                # Send batch for processing
                batch_request = {
                    "session_id": session_id,
                    "data": batch_data.to_dict("records"),
                    "batch_metadata": {
                        "batch_number": i + 1,
                        "source": "test_generator",
                        "expected_anomaly_rate": anomaly_rate,
                    },
                }

                batch_response = app_client.post(
                    "/api/streaming/process-batch", json=batch_request
                )
                assert batch_response.status_code == 200
                batch_result = batch_response.json()
                batch_results.append(batch_result)

                # Verify batch processing results
                assert "batch_id" in batch_result
                assert "anomalies_detected" in batch_result
                assert "anomaly_rate" in batch_result
                assert "processing_time" in batch_result

                # Verify anomaly rate increases with injected anomalies
                detected_rate = batch_result["anomaly_rate"]
                assert 0 <= detected_rate <= 0.5  # Reasonable bounds

            # Get streaming session summary
            summary_response = app_client.get(f"/api/streaming/{session_id}/summary")
            assert summary_response.status_code == 200
            summary_result = summary_response.json()

            assert "total_batches" in summary_result
            assert "total_samples" in summary_result
            assert "overall_anomaly_rate" in summary_result
            assert "performance_metrics" in summary_result
            assert summary_result["total_batches"] == 5

            # Stop streaming session
            stop_response = app_client.post(f"/api/streaming/{session_id}/stop")
            assert stop_response.status_code == 200

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_alert_system_workflow(self, app_client, streaming_data_generator):
        """Test comprehensive alerting system."""
        # Setup detector for alerting
        baseline_data = streaming_data_generator(300, 0.02)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            baseline_data.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            # Upload and setup detector
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("alert_baseline.csv", file, "text/csv")},
                    data={"name": "Alert Baseline"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            detector_data = {
                "name": "Alert Detector",
                "algorithm_name": "LocalOutlierFactor",
                "parameters": {"contamination": 0.05, "n_neighbors": 20},
            }

            create_response = app_client.post("/api/detectors/", json=detector_data)
            assert create_response.status_code == 200
            detector_id = create_response.json()["id"]

            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train", json={"dataset_id": dataset_id}
            )
            assert train_response.status_code == 200

            # Configure alert rules
            alert_rules = [
                {
                    "rule_name": "High Anomaly Rate",
                    "condition": "anomaly_rate > 0.15",
                    "severity": "high",
                    "channels": ["email", "webhook"],
                    "cooldown_minutes": 5,
                },
                {
                    "rule_name": "Consecutive Anomalies",
                    "condition": "consecutive_anomalies >= 3",
                    "severity": "medium",
                    "channels": ["log", "webhook"],
                    "cooldown_minutes": 2,
                },
                {
                    "rule_name": "High Anomaly Score",
                    "condition": "max_anomaly_score > 0.9",
                    "severity": "low",
                    "channels": ["log"],
                    "cooldown_minutes": 1,
                },
            ]

            alert_config = {
                "detector_id": detector_id,
                "alert_name": "Production Alerts",
                "rules": alert_rules,
                "global_settings": {
                    "email_recipients": ["admin@example.com"],
                    "webhook_url": "https://example.com/webhook",
                    "log_level": "INFO",
                },
            }

            alert_setup_response = app_client.post(
                "/api/alerts/setup", json=alert_config
            )
            assert alert_setup_response.status_code == 200
            alert_result = alert_setup_response.json()

            assert "alert_system_id" in alert_result
            assert "active_rules" in alert_result
            alert_system_id = alert_result["alert_system_id"]

            # Simulate conditions that trigger alerts
            test_scenarios = [
                {
                    "name": "Normal Operation",
                    "data": streaming_data_generator(100, 0.03),
                    "expected_alerts": 0,
                },
                {
                    "name": "High Anomaly Rate",
                    "data": streaming_data_generator(100, 0.20),  # 20% anomalies
                    "expected_alerts": 1,  # Should trigger high anomaly rate alert
                },
                {
                    "name": "Extreme Outliers",
                    "data": pd.DataFrame(
                        {
                            "feature_1": [10, 15, 20],  # Extreme values
                            "feature_2": [10, 15, 20],
                            "timestamp": [time.time() + i for i in range(3)],
                            "batch_id": ["extreme_batch"] * 3,
                        }
                    ),
                    "expected_alerts": 1,  # Should trigger high score alert
                },
            ]

            for scenario in test_scenarios:
                scenario_data = scenario["data"]

                # Process scenario data
                detection_request = {
                    "detector_id": detector_id,
                    "data": scenario_data.to_dict("records"),
                    "alert_system_id": alert_system_id,
                    "scenario_name": scenario["name"],
                }

                detection_response = app_client.post(
                    "/api/detection/process-with-alerts", json=detection_request
                )
                assert detection_response.status_code == 200
                detection_result = detection_response.json()

                # Verify detection results
                assert "anomalies" in detection_result
                assert "alerts_triggered" in detection_result
                assert "alert_summary" in detection_result

                # Check alert triggering
                alerts_triggered = detection_result["alerts_triggered"]
                if scenario["expected_alerts"] > 0:
                    assert len(alerts_triggered) >= scenario["expected_alerts"]

                    # Verify alert structure
                    for alert in alerts_triggered:
                        assert "rule_name" in alert
                        assert "severity" in alert
                        assert "timestamp" in alert
                        assert "details" in alert

            # Get alert history
            history_response = app_client.get(f"/api/alerts/{alert_system_id}/history")
            assert history_response.status_code == 200
            history_result = history_response.json()

            assert "alerts" in history_result
            assert "summary" in history_result
            assert (
                len(history_result["alerts"]) > 0
            )  # Should have some alerts from scenarios

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_model_drift_monitoring(self, app_client, streaming_data_generator):
        """Test model drift detection and monitoring."""
        # Create initial training data
        initial_data = streaming_data_generator(1000, 0.02)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            initial_data.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            # Upload initial dataset
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("drift_initial.csv", file, "text/csv")},
                    data={"name": "Drift Monitoring Initial"},
                )
            assert upload_response.status_code == 200
            initial_dataset_id = upload_response.json()["id"]

            # Create and train detector
            detector_data = {
                "name": "Drift Monitor Detector",
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.05, "random_state": 42},
            }

            create_response = app_client.post("/api/detectors/", json=detector_data)
            assert create_response.status_code == 200
            detector_id = create_response.json()["id"]

            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train",
                json={"dataset_id": initial_dataset_id},
            )
            assert train_response.status_code == 200

            # Set up drift monitoring
            drift_config = {
                "detector_id": detector_id,
                "baseline_dataset_id": initial_dataset_id,
                "drift_detection_methods": ["ks_test", "jensen_shannon", "psi"],
                "monitoring_windows": {
                    "short_term": 100,  # samples
                    "medium_term": 500,
                    "long_term": 1000,
                },
                "thresholds": {"ks_test": 0.05, "jensen_shannon": 0.1, "psi": 0.25},
                "alert_on_drift": True,
            }

            drift_setup_response = app_client.post(
                "/api/monitoring/drift/setup", json=drift_config
            )
            assert drift_setup_response.status_code == 200
            drift_result = drift_setup_response.json()

            assert "drift_monitor_id" in drift_result
            drift_monitor_id = drift_result["drift_monitor_id"]

            # Simulate data over time with gradual drift
            time_periods = [
                {"name": "Stable Period", "shift": [0, 0], "scale": [1, 1]},
                {"name": "Slight Drift", "shift": [0.5, 0.2], "scale": [1.1, 1]},
                {"name": "Moderate Drift", "shift": [1.0, 0.5], "scale": [1.2, 1.1]},
                {"name": "Significant Drift", "shift": [2.0, 1.0], "scale": [1.5, 1.3]},
            ]

            drift_reports = []

            for period in time_periods:
                # Generate data with specified drift
                drift_data = streaming_data_generator(200, 0.03)

                # Apply drift transformation
                drift_data["feature_1"] += period["shift"][0]
                drift_data["feature_2"] += period["shift"][1]
                drift_data["feature_1"] *= period["scale"][0]
                drift_data["feature_2"] *= period["scale"][1]

                # Submit data for drift analysis
                drift_analysis_request = {
                    "drift_monitor_id": drift_monitor_id,
                    "new_data": drift_data.to_dict("records"),
                    "period_name": period["name"],
                }

                drift_analysis_response = app_client.post(
                    "/api/monitoring/drift/analyze", json=drift_analysis_request
                )
                assert drift_analysis_response.status_code == 200
                drift_analysis_result = drift_analysis_response.json()

                drift_reports.append(
                    {"period": period["name"], "results": drift_analysis_result}
                )

                # Verify drift analysis results
                assert "drift_detected" in drift_analysis_result
                assert "drift_scores" in drift_analysis_result
                assert "statistical_tests" in drift_analysis_result
                assert "recommendations" in drift_analysis_result

                # Check that drift detection becomes more pronounced over time
                drift_scores = drift_analysis_result["drift_scores"]
                for method, score in drift_scores.items():
                    assert 0 <= score <= 1  # Scores should be normalized

            # Verify drift progression
            stable_score = drift_reports[0]["results"]["drift_scores"]["jensen_shannon"]
            significant_score = drift_reports[-1]["results"]["drift_scores"][
                "jensen_shannon"
            ]
            assert significant_score > stable_score  # Drift should increase

            # Get comprehensive drift report
            report_response = app_client.get(
                f"/api/monitoring/drift/{drift_monitor_id}/report"
            )
            assert report_response.status_code == 200
            report_result = report_response.json()

            assert "drift_timeline" in report_result
            assert "overall_drift_trend" in report_result
            assert "recommendations" in report_result
            assert "retrain_suggestion" in report_result

        finally:
            Path(dataset_file).unlink(missing_ok=True)

    def test_performance_monitoring_workflow(
        self, app_client, streaming_data_generator
    ):
        """Test comprehensive performance monitoring."""
        # Setup baseline
        baseline_data = streaming_data_generator(500, 0.02)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            baseline_data.to_csv(f.name, index=False)
            dataset_file = f.name

        try:
            # Upload and setup detector
            with open(dataset_file, "rb") as file:
                upload_response = app_client.post(
                    "/api/datasets/upload",
                    files={"file": ("performance_baseline.csv", file, "text/csv")},
                    data={"name": "Performance Baseline"},
                )
            assert upload_response.status_code == 200
            dataset_id = upload_response.json()["id"]

            detector_data = {
                "name": "Performance Monitor",
                "algorithm_name": "IsolationForest",
                "parameters": {"contamination": 0.05},
            }

            create_response = app_client.post("/api/detectors/", json=detector_data)
            assert create_response.status_code == 200
            detector_id = create_response.json()["id"]

            train_response = app_client.post(
                f"/api/detectors/{detector_id}/train", json={"dataset_id": dataset_id}
            )
            assert train_response.status_code == 200

            # Set up performance monitoring
            perf_config = {
                "detector_id": detector_id,
                "metrics_to_track": [
                    "inference_time",
                    "throughput",
                    "memory_usage",
                    "cpu_usage",
                    "accuracy_proxy",
                    "false_positive_rate",
                ],
                "performance_thresholds": {
                    "max_inference_time_ms": 100,
                    "min_throughput_per_sec": 1000,
                    "max_memory_mb": 500,
                    "max_cpu_percent": 80,
                },
                "monitoring_interval": "1min",
                "retention_period": "7d",
            }

            perf_setup_response = app_client.post(
                "/api/monitoring/performance/setup", json=perf_config
            )
            assert perf_setup_response.status_code == 200
            perf_result = perf_setup_response.json()

            assert "performance_monitor_id" in perf_result
            perf_monitor_id = perf_result["performance_monitor_id"]

            # Run performance benchmark
            benchmark_config = {
                "monitor_id": perf_monitor_id,
                "test_scenarios": [
                    {"name": "Light Load", "batch_size": 10, "num_batches": 10},
                    {"name": "Medium Load", "batch_size": 100, "num_batches": 10},
                    {"name": "Heavy Load", "batch_size": 1000, "num_batches": 5},
                ],
                "measure_detailed_metrics": True,
            }

            benchmark_response = app_client.post(
                "/api/monitoring/performance/benchmark", json=benchmark_config
            )
            assert benchmark_response.status_code == 200
            benchmark_result = benchmark_response.json()

            # Verify benchmark results
            assert "scenario_results" in benchmark_result
            assert "performance_summary" in benchmark_result
            assert "threshold_violations" in benchmark_result

            scenario_results = benchmark_result["scenario_results"]
            assert len(scenario_results) == 3  # Three test scenarios

            for scenario in scenario_results:
                assert "scenario_name" in scenario
                assert "metrics" in scenario
                assert "performance_grade" in scenario

                metrics = scenario["metrics"]
                assert "avg_inference_time_ms" in metrics
                assert "throughput_per_sec" in metrics
                assert "memory_usage_mb" in metrics
                assert "cpu_usage_percent" in metrics

            # Get real-time performance metrics
            realtime_response = app_client.get(
                f"/api/monitoring/performance/{perf_monitor_id}/current"
            )
            assert realtime_response.status_code == 200
            realtime_result = realtime_response.json()

            assert "current_metrics" in realtime_result
            assert "status" in realtime_result
            assert "last_updated" in realtime_result

        finally:
            Path(dataset_file).unlink(missing_ok=True)
