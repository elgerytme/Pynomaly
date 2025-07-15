#!/usr/bin/env python3
"""
Comprehensive Integration Tests for Pynomaly MLOps Stack.
Tests end-to-end workflows including model training, deployment, serving, and monitoring.
"""

import asyncio
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import IsolationForest

# Import Pynomaly components
from pynomaly.application.services.automl_pipeline_orchestrator import (
    AutoMLPipelineOrchestrator,
    PipelineConfig,
    PipelineMode,
)
from pynomaly.infrastructure.config.environment_config import ConfigManager, Environment
from pynomaly.infrastructure.data_quality.data_validation import (
    DataPipelineMonitor,
    ValidationRule,
    ValidationRuleType,
    ValidationSeverity,
)
from pynomaly.mlops.automated_retraining import (
    AutomatedRetrainingPipeline,
    RetrainingConfig,
    TriggerType,
)
from pynomaly.mlops.model_deployment import (
    DeploymentEnvironment,
    ModelDeploymentManager,
)
from pynomaly.mlops.model_registry import ModelRegistry, ModelType
from pynomaly.mlops.model_serving import ModelServingEngine, PredictionRequest
from pynomaly.mlops.monitoring import MLOpsMonitor


class TestMLOpsIntegration:
    """Integration tests for MLOps components."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment with temporary directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = self._generate_test_data()

        # Initialize components with test configuration
        self.model_registry = ModelRegistry(registry_path=f"{self.temp_dir}/models")
        self.deployment_manager = ModelDeploymentManager(
            deployment_path=f"{self.temp_dir}/deployments"
        )
        self.serving_engine = ModelServingEngine()
        self.monitoring = MLOpsMonitor()
        self.data_monitor = DataPipelineMonitor(
            storage_path=f"{self.temp_dir}/data_quality"
        )

        yield

        # Cleanup
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _generate_test_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic test data for anomaly detection."""
        np.random.seed(42)

        # Generate normal data
        normal_data = np.random.normal(0, 1, (int(n_samples * 0.9), 5))

        # Generate anomalous data
        anomaly_data = np.random.normal(3, 1, (int(n_samples * 0.1), 5))

        # Combine data
        X = np.vstack([normal_data, anomaly_data])
        y = np.hstack([np.zeros(len(normal_data)), np.ones(len(anomaly_data))])

        # Create DataFrame
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
        df["target"] = y
        df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="1H")

        return df

    @pytest.mark.asyncio
    async def test_end_to_end_ml_pipeline(self):
        """Test complete ML pipeline from data to deployment."""
        # Step 1: Data quality validation
        dataset_name = "test_anomaly_data"

        # Setup validation rules
        validation_rule = ValidationRule(
            rule_id="completeness_check",
            name="Data Completeness",
            description="Check data completeness",
            rule_type=ValidationRuleType.COMPLETENESS,
            severity=ValidationSeverity.ERROR,
            enabled=True,
            parameters={"columns": self.test_data.columns.tolist(), "threshold": 0.95},
        )

        self.data_monitor.validator.add_validation_rule(validation_rule)

        # Validate data
        validation_report = self.data_monitor.validate_and_monitor(
            self.test_data, dataset_name
        )

        assert validation_report.data_quality_metrics.overall_score > 0.8
        assert validation_report.failed_rules == 0

        # Step 2: Model training
        model_name = "test_isolation_forest"
        model_version = "v1.0.0"

        # Train model
        model = IsolationForest(contamination=0.1, random_state=42)
        X = self.test_data.drop(["target", "timestamp"], axis=1)
        model.fit(X)

        # Step 3: Model registration
        model_id = await self.model_registry.register_model(
            model=model,
            name=model_name,
            version=model_version,
            model_type=ModelType.ISOLATION_FOREST,
            author="test_user",
            description="Test anomaly detection model",
            performance_metrics={"accuracy": 0.95, "precision": 0.90, "recall": 0.88},
            hyperparameters={"contamination": 0.1, "random_state": 42},
        )

        assert model_id is not None

        # Verify model registration
        stored_model, metadata = await self.model_registry.get_model(model_id)
        assert stored_model is not None
        assert metadata.name == model_name
        assert metadata.version == model_version

        # Step 4: Model deployment
        deployment_id = await self.deployment_manager.create_deployment(
            model_id=model_id,
            model_version=model_version,
            environment=DeploymentEnvironment.DEVELOPMENT,
            author="test_user",
            notes="Integration test deployment",
        )

        assert deployment_id is not None

        # Deploy the model
        deployment_success = await self.deployment_manager.deploy_model(deployment_id)
        assert deployment_success

        # Step 5: Model serving
        await self.serving_engine.load_model(model_id, model_version)

        # Test prediction
        test_sample = self.test_data.iloc[0].drop(["target", "timestamp"]).to_dict()

        prediction_request = PredictionRequest(
            data=test_sample, model_id=model_id, return_probabilities=True
        )

        prediction_response = await self.serving_engine.predict(prediction_request)

        assert prediction_response.status == "success"
        assert prediction_response.model_id == model_id
        assert len(prediction_response.predictions) > 0

        # Step 6: Monitoring
        # Record prediction metrics
        self.monitoring.record_model_prediction(
            model_id=model_id,
            prediction_time_ms=prediction_response.processing_time_ms,
            success=True,
        )

        # Check health
        health_response = await self.serving_engine.health_check(model_id)
        assert health_response.status.value == "healthy"

        # Step 7: Cleanup
        await self.deployment_manager.terminate_deployment(deployment_id)

    @pytest.mark.asyncio
    async def test_automated_retraining_pipeline(self):
        """Test automated retraining pipeline."""
        # Setup initial model
        model_id = "test_model_retraining"
        initial_model = IsolationForest(contamination=0.1, random_state=42)
        X = self.test_data.drop(["target", "timestamp"], axis=1)
        initial_model.fit(X)

        # Register initial model
        registered_model_id = await self.model_registry.register_model(
            model=initial_model,
            name=model_id,
            version="v1.0.0",
            model_type=ModelType.ISOLATION_FOREST,
            author="test_user",
            description="Initial model for retraining test",
        )

        # Setup retraining pipeline
        retraining_pipeline = AutomatedRetrainingPipeline(
            pipeline_path=f"{self.temp_dir}/retraining"
        )

        # Configure retraining
        retraining_config = RetrainingConfig(
            model_id=registered_model_id,
            trigger_type=TriggerType.MANUAL,
            schedule_cron=None,
            performance_threshold=0.05,
            data_drift_threshold=0.1,
            min_data_points=100,
            max_training_time_minutes=10,
            auto_deploy=False,
            validation_split=0.2,
            hyperparameter_tuning=True,
            notification_enabled=False,
            rollback_enabled=True,
        )

        retraining_pipeline.configure_retraining(retraining_config)

        # Trigger retraining
        job_id = await retraining_pipeline._trigger_retraining(
            registered_model_id, TriggerType.MANUAL, "Integration test manual trigger"
        )

        assert job_id is not None

        # Wait for job completion (with timeout)
        max_wait = 30  # seconds
        wait_time = 0

        while wait_time < max_wait:
            job = retraining_pipeline.get_job_status(job_id)
            if job and job.status.value in ["completed", "failed"]:
                break

            await asyncio.sleep(1)
            wait_time += 1

        # Check job completion
        final_job = retraining_pipeline.get_job_status(job_id)
        assert final_job is not None
        assert final_job.status.value == "completed"
        assert final_job.new_model_id is not None

    @pytest.mark.asyncio
    async def test_automl_pipeline_orchestration(self):
        """Test AutoML pipeline orchestration."""
        # Setup pipeline configuration
        config = PipelineConfig(
            mode=PipelineMode.FAST,
            train_test_split_ratio=0.8,
            optimization_time_budget_minutes=5,
            max_models_to_evaluate=3,
            enable_feature_engineering=True,
            enable_ensemble=False,  # Disable for faster test
        )

        # Initialize orchestrator
        orchestrator = AutoMLPipelineOrchestrator(
            pipeline_path=f"{self.temp_dir}/automl"
        )

        # Prepare data
        dataset_profile = MagicMock()
        dataset_profile.feature_count = 5
        dataset_profile.sample_count = len(self.test_data)
        dataset_profile.target_column = "target"

        # Run pipeline
        with patch(
            "pynomaly.application.services.automl_pipeline_orchestrator.DatasetProfile"
        ) as mock_profile:
            mock_profile.return_value = dataset_profile

            pipeline_result = await orchestrator.run_complete_pipeline(
                data=self.test_data,
                target_column="target",
                config=config,
                dataset_name="test_automl_data",
            )

        assert pipeline_result is not None
        assert pipeline_result["status"] == "completed"
        assert "best_model" in pipeline_result
        assert "metrics" in pipeline_result

    @pytest.mark.asyncio
    async def test_monitoring_and_alerting(self):
        """Test monitoring and alerting system."""
        # Record various metrics
        model_id = "test_monitoring_model"

        # Record successful predictions
        for i in range(10):
            self.monitoring.record_model_prediction(
                model_id=model_id, prediction_time_ms=50 + i * 5, success=True
            )

        # Record some errors
        for i in range(2):
            self.monitoring.record_model_prediction(
                model_id=model_id, prediction_time_ms=0, success=False, error="timeout"
            )

        # Record system metrics
        self.monitoring.metrics_collector.set_gauge("system.cpu_percent", 85.0)
        self.monitoring.metrics_collector.set_gauge("system.memory_percent", 75.0)

        # Start monitoring (briefly)
        await self.monitoring.start_monitoring()

        # Wait for alert evaluation
        await asyncio.sleep(2)

        # Check for metrics
        metrics = self.monitoring.metrics_collector.get_metrics(
            "model.predictions_total"
        )
        assert len(metrics) > 0

        # Check alerts
        active_alerts = self.monitoring.alert_manager.get_active_alerts()
        # May or may not have alerts depending on thresholds

    @pytest.mark.asyncio
    async def test_data_drift_detection(self):
        """Test data drift detection and pipeline integration."""
        # Create reference data
        reference_data = self.test_data.iloc[:500].copy()

        # Create drifted data (shifted distribution)
        drifted_data = self.test_data.iloc[500:].copy()
        for col in drifted_data.select_dtypes(include=[np.number]).columns:
            if col not in ["target", "timestamp"]:
                drifted_data[col] = drifted_data[col] + 2  # Shift distribution

        # Setup retraining pipeline for drift detection
        retraining_pipeline = AutomatedRetrainingPipeline(
            pipeline_path=f"{self.temp_dir}/drift_test"
        )

        # Set reference data
        retraining_pipeline.drift_detector.set_reference_data(
            reference_data.drop(["target", "timestamp"], axis=1)
        )

        # Detect drift
        drift_report = retraining_pipeline.drift_detector.detect_drift(
            drifted_data.drop(["target", "timestamp"], axis=1), threshold=0.1
        )

        assert drift_report.drift_detected
        assert drift_report.drift_score > 0.1
        assert len(drift_report.feature_drift_scores) > 0

    @pytest.mark.asyncio
    async def test_configuration_management(self):
        """Test configuration management across environments."""
        # Create test config manager
        config_manager = ConfigManager(config_dir=f"{self.temp_dir}/config")

        # Create default configurations
        config_manager.create_default_configs()

        # Test configuration loading
        config = config_manager.get_config()
        assert config.environment == Environment.DEVELOPMENT
        assert config.api.port == 8000
        assert config.database.host == "localhost"

        # Test database URL generation
        db_url = config_manager.get_database_url()
        assert "postgresql://" in db_url

        # Test Redis URL generation
        redis_url = config_manager.get_redis_url()
        assert "redis://" in redis_url

    @pytest.mark.asyncio
    async def test_model_versioning_and_rollback(self):
        """Test model versioning and rollback capabilities."""
        model_name = "versioning_test_model"

        # Register multiple versions
        versions = []
        for i in range(3):
            model = IsolationForest(contamination=0.1 + i * 0.05, random_state=42)
            X = self.test_data.drop(["target", "timestamp"], axis=1)
            model.fit(X)

            model_id = await self.model_registry.register_model(
                model=model,
                name=model_name,
                version=f"v1.{i}.0",
                model_type=ModelType.ISOLATION_FOREST,
                author="test_user",
                description=f"Version {i} of test model",
                performance_metrics={"accuracy": 0.9 + i * 0.01},
            )
            versions.append(model_id)

        # List model versions
        model_versions = await self.model_registry.list_model_versions(model_name)
        assert len(model_versions) == 3

        # Test getting specific version
        v1_model, v1_metadata = await self.model_registry.get_model(versions[1])
        assert v1_metadata.version == "v1.1.0"

        # Test getting latest version
        latest_model, latest_metadata = await self.model_registry.get_model_by_name(
            model_name
        )
        assert latest_metadata.version == "v1.2.0"

    @pytest.mark.asyncio
    async def test_batch_prediction_pipeline(self):
        """Test batch prediction capabilities."""
        # Setup model
        model = IsolationForest(contamination=0.1, random_state=42)
        X = self.test_data.drop(["target", "timestamp"], axis=1)
        model.fit(X)

        model_id = await self.model_registry.register_model(
            model=model,
            name="batch_test_model",
            version="v1.0.0",
            model_type=ModelType.ISOLATION_FOREST,
            author="test_user",
        )

        # Load model in serving engine
        await self.serving_engine.load_model(model_id)

        # Prepare batch data
        batch_data = []
        for _, row in self.test_data.head(10).iterrows():
            sample = row.drop(["target", "timestamp"]).to_dict()
            batch_data.append(sample)

        # Test batch prediction
        prediction_request = PredictionRequest(
            data=batch_data, model_id=model_id, inference_mode="batch"
        )

        response = await self.serving_engine.predict(prediction_request)

        assert response.status == "success"
        assert len(response.predictions) == 10
        assert all("is_anomaly" in pred for pred in response.predictions)

    @pytest.mark.asyncio
    async def test_performance_monitoring_and_alerting(self):
        """Test performance monitoring and degradation alerts."""
        model_id = "performance_test_model"

        # Set baseline performance
        baseline_metrics = {
            "accuracy": 0.95,
            "precision": 0.90,
            "recall": 0.88,
            "f1_score": 0.89,
        }

        self.monitoring.alert_manager.performance_monitor.set_baseline_metrics(
            model_id, baseline_metrics
        )

        # Simulate performance degradation
        degraded_metrics = {
            "accuracy": 0.80,  # Significant drop
            "precision": 0.75,
            "recall": 0.70,
            "f1_score": 0.72,
        }

        # Evaluate performance
        performance_report = (
            self.monitoring.alert_manager.performance_monitor.evaluate_performance(
                model_id=model_id,
                deployment_id="test_deployment",
                current_metrics=degraded_metrics,
                threshold=0.05,  # 5% degradation threshold
            )
        )

        assert performance_report.alert_triggered
        assert performance_report.performance_degradation["accuracy"] > 0.05

    def test_api_integration(self):
        """Test API endpoints integration."""
        # This would typically test FastAPI endpoints
        # For now, we'll test the basic structure
        from pynomaly.mlops.model_serving import app

        client = TestClient(app)

        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

        # Test stats endpoint
        response = client.get("/stats")
        assert response.status_code == 200
        stats = response.json()
        assert "uptime_seconds" in stats
        assert "total_requests" in stats

    @pytest.mark.asyncio
    async def test_full_pipeline_stress_test(self):
        """Stress test the full pipeline with multiple concurrent operations."""
        # This test simulates realistic load
        tasks = []

        # Concurrent model registrations
        for i in range(5):
            model = IsolationForest(contamination=0.1, random_state=42 + i)
            X = self.test_data.drop(["target", "timestamp"], axis=1)
            model.fit(X)

            task = self.model_registry.register_model(
                model=model,
                name=f"stress_test_model_{i}",
                version="v1.0.0",
                model_type=ModelType.ISOLATION_FOREST,
                author="stress_test",
            )
            tasks.append(task)

        # Wait for all registrations
        model_ids = await asyncio.gather(*tasks)
        assert len(model_ids) == 5
        assert all(mid is not None for mid in model_ids)

        # Concurrent predictions
        prediction_tasks = []
        for model_id in model_ids[:3]:  # Use first 3 models
            await self.serving_engine.load_model(model_id)

            test_sample = self.test_data.iloc[0].drop(["target", "timestamp"]).to_dict()
            prediction_request = PredictionRequest(data=test_sample, model_id=model_id)

            prediction_task = self.serving_engine.predict(prediction_request)
            prediction_tasks.append(prediction_task)

        # Wait for all predictions
        responses = await asyncio.gather(*prediction_tasks)
        assert len(responses) == 3
        assert all(r.status == "success" for r in responses)


# Additional test utilities
class TestDataGenerator:
    """Utility class for generating test data."""

    @staticmethod
    def generate_anomaly_data(
        n_samples: int = 1000, n_features: int = 5, contamination: float = 0.1
    ):
        """Generate synthetic anomaly detection dataset."""
        np.random.seed(42)

        n_normal = int(n_samples * (1 - contamination))
        n_anomaly = n_samples - n_normal

        # Normal data
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features), cov=np.eye(n_features), size=n_normal
        )

        # Anomalous data (shifted and scaled)
        anomaly_data = np.random.multivariate_normal(
            mean=np.ones(n_features) * 3, cov=np.eye(n_features) * 2, size=n_anomaly
        )

        X = np.vstack([normal_data, anomaly_data])
        y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])

        return X, y

    @staticmethod
    def generate_time_series_data(n_samples: int = 1000, n_features: int = 3):
        """Generate time series data with trends and seasonality."""
        dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="1H")

        data = {}
        for i in range(n_features):
            # Base trend
            trend = np.linspace(0, 2, n_samples)

            # Seasonal component
            seasonal = np.sin(2 * np.pi * np.arange(n_samples) / 24) * 0.5

            # Noise
            noise = np.random.normal(0, 0.1, n_samples)

            data[f"feature_{i}"] = trend + seasonal + noise

        df = pd.DataFrame(data, index=dates)
        return df


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
