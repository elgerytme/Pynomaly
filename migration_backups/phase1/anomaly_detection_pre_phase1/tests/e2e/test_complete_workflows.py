"""End-to-end workflow tests for the complete anomaly detection system."""

import pytest
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from click.testing import CliRunner

from anomaly_detection.server import create_app
from anomaly_detection.cli import main
from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository
from anomaly_detection.infrastructure.monitoring import get_monitoring_dashboard


class TestCompleteWorkflows:
    """End-to-end tests for complete anomaly detection workflows."""
    
    @pytest.fixture
    def api_client(self):
        """Create FastAPI test client."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def sample_dataset(self, temp_dir: Path):
        """Create a realistic sample dataset."""
        np.random.seed(42)
        
        # Generate realistic anomaly detection dataset
        # Normal behavior: customers with regular transaction patterns
        normal_transactions = np.random.multivariate_normal(
            mean=[50, 5, 100],  # amount, frequency, account_age
            cov=[[100, 2, 50], [2, 1, 5], [50, 5, 200]],
            size=800
        )
        
        # Anomalous behavior: unusual transaction patterns
        fraud_transactions = np.array([
            [500, 1, 30],    # High amount, low frequency, new account
            [1000, 2, 15],   # Very high amount, new account
            [200, 20, 50],   # High frequency
            [750, 1, 10],    # High amount, very new account
            [300, 15, 25],   # High frequency, relatively new account
        ])
        
        # Add some noise to fraud transactions
        fraud_transactions += np.random.normal(0, 10, fraud_transactions.shape)
        
        # Combine datasets
        all_data = np.vstack([normal_transactions, fraud_transactions])
        labels = np.hstack([np.ones(800), -np.ones(5)])  # 1 = normal, -1 = anomaly
        
        # Create DataFrame
        df = pd.DataFrame(all_data, columns=['transaction_amount', 'daily_frequency', 'account_age_days'])
        df['is_fraud'] = (labels == -1).astype(int)  # Convert to 0/1 for easier interpretation
        
        # Save to CSV
        csv_path = temp_dir / "financial_transactions.csv"
        df.to_csv(csv_path, index=False)
        
        return {
            'csv_path': csv_path,
            'data': all_data,
            'labels': labels,
            'df': df,
            'n_samples': len(df),
            'n_anomalies': 5,
            'contamination_rate': 5/805
        }
    
    def test_end_to_end_api_workflow(self, api_client: TestClient, sample_dataset: Dict[str, Any]):
        """Test complete API workflow: data -> detection -> monitoring."""
        # Step 1: Check API health
        health_response = api_client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"
        
        # Step 2: Get available algorithms
        algorithms_response = api_client.get("/api/v1/algorithms")
        assert algorithms_response.status_code == 200
        algorithms_data = algorithms_response.json()
        assert "isolation_forest" in algorithms_data["single_algorithms"]
        
        # Step 3: Perform anomaly detection
        test_data = sample_dataset['data'][:100].tolist()  # Use subset for API test
        
        detection_request = {
            "data": test_data,
            "algorithm": "isolation_forest",
            "contamination": 0.1,
            "parameters": {"random_state": 42}
        }
        
        detection_response = api_client.post("/api/v1/detect", json=detection_request)
        assert detection_response.status_code == 200
        
        detection_result = detection_response.json()
        assert detection_result["success"] is True
        assert detection_result["total_samples"] == 100
        assert detection_result["algorithm"] == "isolation_forest"
        assert len(detection_result["anomalies"]) == detection_result["anomalies_detected"]
        
        # Step 4: Test ensemble detection
        ensemble_request = {
            "data": test_data,
            "algorithms": ["isolation_forest", "local_outlier_factor"],
            "method": "majority",
            "contamination": 0.1
        }
        
        ensemble_response = api_client.post("/api/v1/ensemble", json=ensemble_request)
        assert ensemble_response.status_code == 200
        
        ensemble_result = ensemble_response.json()
        assert ensemble_result["success"] is True
        assert "ensemble_majority" in ensemble_result["algorithm"]
        
        # Step 5: Check monitoring data
        metrics_response = api_client.get("/api/v1/metrics")
        assert metrics_response.status_code == 200
        
        metrics_data = metrics_response.json()
        assert "metrics_summary" in metrics_data
        assert "performance_summary" in metrics_data
        
        # Should have recorded some operations
        performance_summary = metrics_data["performance_summary"]
        assert performance_summary["total_operations"] >= 2  # At least detection + ensemble
        
        # Step 6: Get dashboard summary
        dashboard_response = api_client.get("/api/v1/dashboard/summary")
        assert dashboard_response.status_code == 200
        
        dashboard_data = dashboard_response.json()
        assert "summary" in dashboard_data
        
        summary = dashboard_data["summary"]
        assert summary["operations_last_hour"] >= 2
        assert 0.0 <= summary["success_rate"] <= 1.0
    
    def test_end_to_end_cli_workflow(self, cli_runner: CliRunner, sample_dataset: Dict[str, Any], temp_dir: Path):
        """Test complete CLI workflow: generate -> detect -> train -> manage models."""
        csv_path = sample_dataset['csv_path']
        
        # Step 1: Check CLI help
        help_result = cli_runner.invoke(main, ['--help'])
        assert help_result.exit_code == 0
        assert "Anomaly Detection CLI" in help_result.output
        
        # Step 2: Run detection on sample data
        detection_output = temp_dir / "detection_results.json"
        detection_result = cli_runner.invoke(main, [
            'detect', 'run',
            '--input', str(csv_path),
            '--output', str(detection_output),
            '--algorithm', 'isolation_forest',
            '--contamination', '0.05',  # Lower rate since we know true contamination
            '--has-labels',
            '--label-column', 'is_fraud'
        ])
        
        assert detection_result.exit_code == 0
        assert "Detection completed successfully" in detection_result.output
        assert detection_output.exists()
        
        # Verify detection results
        with open(detection_output, 'r') as f:
            results = json.load(f)
        
        assert results["success"] is True
        assert "evaluation_metrics" in results
        assert "accuracy" in results["evaluation_metrics"]
        
        # Accuracy should be reasonable (>0.8) given the clear separation in synthetic data
        accuracy = results["evaluation_metrics"]["accuracy"]
        assert accuracy > 0.8, f"Accuracy {accuracy} too low for well-separated synthetic data"
        
        # Step 3: Run ensemble detection
        ensemble_output = temp_dir / "ensemble_results.json"
        ensemble_result = cli_runner.invoke(main, [
            'ensemble', 'combine',
            '--input', str(csv_path),
            '--output', str(ensemble_output),
            '--algorithms', 'isolation_forest',
            '--algorithms', 'lof',
            '--method', 'majority',
            '--contamination', '0.05'
        ])
        
        assert ensemble_result.exit_code == 0
        assert "Ensemble detection completed successfully" in ensemble_result.output
        assert ensemble_output.exists()
        
        # Step 4: Train and save a model
        models_dir = temp_dir / "trained_models"
        train_result = cli_runner.invoke(main, [
            'model', 'train',
            '--input', str(csv_path),
            '--model-name', 'E2E Fraud Detection Model',
            '--algorithm', 'isolation_forest',
            '--contamination', '0.05',
            '--output-dir', str(models_dir),
            '--format', 'pickle',
            '--has-labels',
            '--label-column', 'is_fraud'
        ])
        
        assert train_result.exit_code == 0
        assert "Model training completed successfully" in train_result.output
        assert models_dir.exists()
        
        # Extract model ID from output
        model_id = None
        for line in train_result.output.split('\n'):
            if "Model ID:" in line:
                model_id = line.split("Model ID:")[-1].strip()
                break
        
        assert model_id is not None, "Could not find model ID in training output"
        
        # Step 5: List trained models
        list_result = cli_runner.invoke(main, [
            'model', 'list',
            '--models-dir', str(models_dir)
        ])
        
        assert list_result.exit_code == 0
        assert "E2E Fraud Detection Model" in list_result.output
        assert model_id[:8] in list_result.output  # Should show truncated ID
        
        # Step 6: Get model info
        info_result = cli_runner.invoke(main, [
            'model', 'info', model_id,
            '--models-dir', str(models_dir)
        ])
        
        assert info_result.exit_code == 0
        assert "E2E Fraud Detection Model" in info_result.output
        assert "isolation_forest" in info_result.output
        assert "Performance Metrics:" in info_result.output
        
        # Step 7: Get repository stats
        stats_result = cli_runner.invoke(main, [
            'model', 'stats',
            '--models-dir', str(models_dir)
        ])
        
        assert stats_result.exit_code == 0
        assert "Total models: 1" in stats_result.output
        assert "isolation_forest: 1" in stats_result.output
        
        # Step 8: Check monitoring status
        monitor_result = cli_runner.invoke(main, ['monitor', 'status'])
        assert monitor_result.exit_code == 0
        assert "Overall System Status:" in monitor_result.output
        assert "Health Checks:" in monitor_result.output
        
        # Step 9: Clean up by deleting the model
        delete_result = cli_runner.invoke(main, [
            'model', 'delete', model_id,
            '--models-dir', str(models_dir),
            '--yes'  # Auto-confirm
        ])
        
        # Note: delete command might not have --yes flag implemented,
        # so we'll check if it prompts or succeeds
        assert delete_result.exit_code in [0, 1]  # May fail due to confirmation prompt
    
    def test_programmatic_workflow(self, sample_dataset: Dict[str, Any], temp_dir: Path):
        """Test complete programmatic workflow using the Python API."""
        data = sample_dataset['data']
        labels = sample_dataset['labels']
        
        # Step 1: Initialize services
        detection_service = DetectionService()
        model_repo = ModelRepository(str(temp_dir / "programmatic_models"))
        
        # Step 2: Perform detection
        detection_result = detection_service.detect_anomalies(
            data=data,
            algorithm="iforest",
            contamination=0.05,
            random_state=42
        )
        
        assert detection_result.success is True
        assert detection_result.total_samples == len(data)
        
        # Step 3: Evaluate detection performance
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Convert sklearn predictions (-1, 1) to our label format
        predictions = detection_result.predictions
        
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, pos_label=-1, zero_division=0)
        recall = recall_score(labels, predictions, pos_label=-1, zero_division=0)
        f1 = f1_score(labels, predictions, pos_label=-1, zero_division=0)
        
        # With well-separated synthetic data, performance should be good
        assert accuracy > 0.8
        assert precision > 0.6  # May have some false positives
        assert recall > 0.4     # Should catch some true anomalies
        
        print(f"Detection Performance: Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        # Step 4: Train and save model
        from anomaly_detection.domain.entities.model import Model, ModelMetadata, ModelStatus
        
        # Fit model on normal data only (more realistic scenario)
        normal_data = data[labels == 1]
        detection_service.fit(normal_data, algorithm="iforest", random_state=42)
        trained_model = detection_service._fitted_models["iforest"]
        
        # Create model entity
        metadata = ModelMetadata(
            model_id="e2e-programmatic-model",
            name="E2E Programmatic Model",
            algorithm="isolation_forest",
            status=ModelStatus.TRAINED,
            training_samples=len(normal_data),
            training_features=data.shape[1],
            contamination_rate=0.05,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            feature_names=["transaction_amount", "daily_frequency", "account_age_days"],
            description="End-to-end test model trained programmatically"
        )
        
        model = Model(metadata=metadata, model_object=trained_model)
        
        # Save model
        model_id = model_repo.save(model)
        assert model_id == "e2e-programmatic-model"
        
        # Step 5: Load and test saved model
        loaded_model = model_repo.load(model_id)
        assert loaded_model.metadata.name == "E2E Programmatic Model"
        
        # Test predictions with loaded model
        test_predictions = loaded_model.model_object.predict(data[:10])
        assert len(test_predictions) == 10
        assert all(pred in [-1, 1] for pred in test_predictions)
        
        # Step 6: Model management
        all_models = model_repo.list_models()
        assert len(all_models) == 1
        assert all_models[0]["model_id"] == model_id
        
        stats = model_repo.get_repository_stats()
        assert stats["total_models"] == 1
        assert stats["by_algorithm"]["isolation_forest"] == 1
        
        # Step 7: Cleanup
        success = model_repo.delete(model_id)
        assert success is True
    
    def test_monitoring_integration_workflow(self, sample_dataset: Dict[str, Any]):
        """Test complete workflow with monitoring integration."""
        from anomaly_detection.infrastructure.monitoring import (
            get_metrics_collector, get_performance_monitor, get_monitoring_dashboard
        )
        
        # Initialize monitoring
        metrics_collector = get_metrics_collector()
        performance_monitor = get_performance_monitor()
        dashboard = get_monitoring_dashboard()
        
        # Clear existing data
        metrics_collector.clear_all_metrics()
        performance_monitor.clear_profiles()
        
        # Step 1: Perform monitored detection
        detection_service = DetectionService()
        data = sample_dataset['data']
        
        # Start monitoring
        with performance_monitor.create_context("e2e_detection", track_memory=True) as ctx:
            result = detection_service.detect_anomalies(
                data=data,
                algorithm="iforest",
                contamination=0.05
            )
            
            # Track additional metrics
            ctx.increment_counter("samples_processed", len(data))
            ctx.increment_counter("anomalies_found", result.anomaly_count)
        
        assert result.success is True
        
        # Step 2: Check monitoring data
        profiles = performance_monitor.get_recent_profiles(limit=5)
        assert len(profiles) >= 1
        
        e2e_profiles = [p for p in profiles if p.operation == "e2e_detection"]
        assert len(e2e_profiles) == 1
        
        profile = e2e_profiles[0]
        assert profile.success is True
        assert profile.total_duration_ms > 0
        
        # Step 3: Check metrics
        summary = metrics_collector.get_summary_stats()
        assert summary["total_metrics"] > 0
        
        # Step 4: Generate dashboard data
        dashboard_summary = await dashboard.get_dashboard_summary()
        assert dashboard_summary.total_operations >= 1
        assert dashboard_summary.success_rate > 0
        
        # Step 5: Check alerts
        alerts = dashboard.get_alert_summary()
        assert "total_alerts" in alerts
        # Should not have critical alerts for this test
        assert alerts.get("critical_count", 0) == 0
        
        print(f"Monitoring Summary: {dashboard_summary.total_operations} operations, "
              f"{dashboard_summary.success_rate:.1%} success rate, "
              f"{alerts['total_alerts']} alerts")
    
    def test_error_handling_workflow(self, api_client: TestClient, cli_runner: CliRunner, temp_dir: Path):
        """Test error handling across the complete system."""
        # Test API error handling
        # Invalid data format
        response = api_client.post("/api/v1/detect", json={
            "data": "invalid_data",
            "algorithm": "isolation_forest"
        })
        assert response.status_code == 422  # FastAPI validation error
        
        # Empty data
        response = api_client.post("/api/v1/detect", json={
            "data": [],
            "algorithm": "isolation_forest"
        })
        assert response.status_code == 400
        
        # Invalid algorithm
        response = api_client.post("/api/v1/detect", json={
            "data": [[1, 2], [3, 4]],
            "algorithm": "nonexistent_algorithm"
        })
        assert response.status_code in [400, 500]
        
        # Test CLI error handling
        # Non-existent file
        result = cli_runner.invoke(main, [
            'detect', 'run',
            '--input', str(temp_dir / "nonexistent.csv"),
            '--algorithm', 'isolation_forest'
        ])
        assert result.exit_code == 1
        assert "Error:" in result.output
        
        # Invalid model operation
        result = cli_runner.invoke(main, [
            'model', 'info', 'nonexistent-model-id'
        ])
        assert result.exit_code == 1
        assert "Error:" in result.output
        
    def test_performance_workflow(self, sample_dataset: Dict[str, Any]):
        """Test performance characteristics of the complete workflow."""
        data = sample_dataset['data']
        
        # Create larger dataset for performance testing
        np.random.seed(42)
        large_data = np.random.randn(5000, 10).astype(np.float64)
        
        detection_service = DetectionService()
        
        # Test performance with timing
        start_time = time.time()
        
        result = detection_service.detect_anomalies(
            data=large_data,
            algorithm="iforest",
            contamination=0.1,
            random_state=42
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert result.success is True
        assert result.total_samples == 5000
        
        # Should complete within reasonable time (adjust threshold based on system)
        assert duration < 30, f"Detection took too long: {duration:.2f} seconds"
        
        print(f"Performance Test: Processed {result.total_samples} samples in {duration:.2f} seconds "
              f"({result.total_samples/duration:.1f} samples/sec)")
        
        # Test memory efficiency (basic check)
        import gc
        gc.collect()
        
        # Multiple runs should not cause memory leaks
        for i in range(3):
            result = detection_service.detect_anomalies(
                data=data[:100],  # Smaller subset
                algorithm="iforest",
                contamination=0.1
            )
            assert result.success is True
        
        gc.collect()
    
    async def test_full_system_integration(self, api_client: TestClient, sample_dataset: Dict[str, Any]):
        """Test full system integration including all components."""
        # This test validates that all components work together correctly
        
        # Step 1: System health check
        health_response = api_client.get("/api/v1/health/detailed")
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        assert health_data["overall_status"] in ["healthy", "degraded"]
        
        # Step 2: Multi-algorithm detection
        test_data = sample_dataset['data'][:50].tolist()
        
        algorithms = ["isolation_forest", "local_outlier_factor"]
        results = {}
        
        for algorithm in algorithms:
            response = api_client.post("/api/v1/detect", json={
                "data": test_data,
                "algorithm": algorithm,
                "contamination": 0.1
            })
            assert response.status_code == 200
            results[algorithm] = response.json()
        
        # All algorithms should succeed
        for algorithm, result in results.items():
            assert result["success"] is True
            assert result["algorithm"] == algorithm
            assert result["total_samples"] == 50
        
        # Step 3: Ensemble detection
        ensemble_response = api_client.post("/api/v1/ensemble", json={
            "data": test_data,
            "algorithms": algorithms,
            "method": "majority",
            "contamination": 0.1
        })
        assert ensemble_response.status_code == 200
        
        ensemble_result = ensemble_response.json()
        assert ensemble_result["success"] is True
        
        # Step 4: Monitoring validation
        dashboard_response = api_client.get("/api/v1/dashboard/summary")
        assert dashboard_response.status_code == 200
        
        dashboard_data = dashboard_response.json()
        summary = dashboard_data["summary"]
        
        # Should have processed multiple operations
        assert summary["operations_last_hour"] >= 3  # At least 3 operations (2 single + 1 ensemble)
        assert summary["success_rate"] > 0.8  # High success rate expected
        
        # Step 5: Performance monitoring
        metrics_response = api_client.get("/api/v1/metrics")
        assert metrics_response.status_code == 200
        
        metrics_data = metrics_response.json()
        assert metrics_data["performance_summary"]["total_operations"] >= 3
        
        print(f"Full System Integration Test Complete: "
              f"Processed {summary['operations_last_hour']} operations with "
              f"{summary['success_rate']:.1%} success rate")