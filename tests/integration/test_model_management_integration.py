"""Integration tests for model management features."""

import asyncio
import shutil
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult, Detector, Model
from pynomaly.domain.value_objects import AnomalyScore, AnomalyType, SemanticVersion
from pynomaly.features.model_management import (
    AutoMLPipeline,
    DeploymentConfig,
    ModelDeployment,
    ModelMonitoring,
    ModelRegistry,
    ModelStatus,
    ModelVersioning,
    PerformanceMetrics,
    get_model_registry,
)


class MockMLModel:
    """Mock ML model for testing."""

    def __init__(self, name: str = "mock_model"):
        self.name = name
        self.is_fitted = False
        self.model_params = {"param1": 1.0, "param2": "test"}

    def fit(self, X, y=None):
        """Mock fit method."""
        self.is_fitted = True
        return self

    def predict(self, X):
        """Mock predict method."""
        return np.random.random(len(X))

    def get_params(self):
        """Get model parameters."""
        return self.model_params


class MockDetector(Detector):
    """Mock detector for testing."""

    def __init__(self, algorithm: str = "mock_detector"):
        self.algorithm = algorithm
        self.is_fitted = False
        self.model = MockMLModel()

    def fit(self, dataset: Dataset) -> None:
        """Mock fit method."""
        self.model.fit(dataset.data.values)
        self.is_fitted = True

    def predict(self, dataset: Dataset) -> DetectionResult:
        """Mock predict method."""
        scores = self.model.predict(dataset.data.values)
        anomalies = []

        for i, score in enumerate(scores):
            if score > 0.7:  # Threshold for anomaly
                anomaly = Anomaly(
                    id=f"anomaly_{i}",
                    score=AnomalyScore(score),
                    type=AnomalyType.POINT,
                    timestamp=datetime.now(),
                    data_point=dataset.data.iloc[i].to_dict(),
                )
                anomalies.append(anomaly)

        return DetectionResult(
            anomalies=anomalies,
            threshold=0.7,
            metadata={
                "algorithm": self.algorithm,
                "execution_time_ms": 50.0,
                "model_version": "1.0.0",
            },
        )


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    data = pd.DataFrame(
        {
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
            "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
        }
    )

    return Dataset(
        name="test_dataset",
        data=data,
        description="Test dataset for model management integration",
    )


@pytest.fixture
def sample_model():
    """Create sample model for testing."""
    return Model(
        name="test_model",
        algorithm="isolation_forest",
        parameters={"contamination": 0.1, "n_estimators": 100},
        description="Test model for integration testing",
    )


@pytest.fixture
def sample_detector():
    """Create sample detector for testing."""
    return MockDetector("test_algorithm")


@pytest.mark.asyncio
class TestModelRegistryIntegration:
    """Integration tests for model registry."""

    async def test_model_registry_lifecycle(self, sample_model, temp_model_dir):
        """Test complete model registry lifecycle."""
        registry = ModelRegistry(storage_path=temp_model_dir)

        # Register model
        metadata = await registry.register_model(
            sample_model, name="test_model_v1", version=SemanticVersion("1.0.0")
        )

        # Verify registration
        assert metadata.name == "test_model_v1"
        assert metadata.version.major == 1
        assert metadata.status == ModelStatus.REGISTERED

        # List models
        models = await registry.list_models()
        assert len(models) == 1
        assert models[0].name == "test_model_v1"

        # Get model
        retrieved_model = await registry.get_model(
            "test_model_v1", SemanticVersion("1.0.0")
        )
        assert retrieved_model is not None
        assert retrieved_model.name == sample_model.name

        # Update model metadata
        updated = await registry.update_model_metadata(
            "test_model_v1", SemanticVersion("1.0.0"), status=ModelStatus.VALIDATED
        )
        assert updated

        # Verify update
        updated_metadata = await registry.get_model_metadata(
            "test_model_v1", SemanticVersion("1.0.0")
        )
        assert updated_metadata.status == ModelStatus.VALIDATED

        # Delete model
        deleted = await registry.delete_model("test_model_v1", SemanticVersion("1.0.0"))
        assert deleted

        # Verify deletion
        models_after_delete = await registry.list_models()
        assert len(models_after_delete) == 0

    async def test_model_versioning_integration(self, sample_model, temp_model_dir):
        """Test model versioning integration."""
        registry = ModelRegistry(storage_path=temp_model_dir)
        versioning = ModelVersioning(registry)

        # Create initial version
        v1_metadata = await versioning.create_version(
            sample_model,
            name="versioned_model",
            version=SemanticVersion("1.0.0"),
            description="Initial version",
        )
        assert v1_metadata.version.major == 1

        # Create patch version
        updated_model = Model(
            name="versioned_model",
            algorithm="isolation_forest",
            parameters={
                "contamination": 0.05,
                "n_estimators": 100,
            },  # Updated parameter
            description="Updated model with different contamination",
        )

        v1_1_metadata = await versioning.create_version(
            updated_model,
            name="versioned_model",
            version=SemanticVersion("1.1.0"),
            description="Bug fix version",
        )
        assert v1_1_metadata.version.minor == 1

        # Get version history
        history = await versioning.get_version_history("versioned_model")
        assert len(history) == 2
        assert any(v.version.patch == 0 for v in history)
        assert any(v.version.minor == 1 for v in history)

        # Get latest version
        latest = await versioning.get_latest_version("versioned_model")
        assert latest.version.minor == 1

        # Compare versions
        comparison = await versioning.compare_versions(
            "versioned_model", SemanticVersion("1.0.0"), SemanticVersion("1.1.0")
        )
        assert comparison["parameter_changes"]["contamination"]["old"] == 0.1
        assert comparison["parameter_changes"]["contamination"]["new"] == 0.05

    async def test_model_deployment_integration(
        self, sample_model, sample_detector, temp_model_dir
    ):
        """Test model deployment integration."""
        registry = ModelRegistry(storage_path=temp_model_dir)
        deployment = ModelDeployment(registry)

        # Register model first
        metadata = await registry.register_model(
            sample_model, name="deployment_model", version=SemanticVersion("1.0.0")
        )

        # Create deployment config
        config = DeploymentConfig(
            environment="testing",
            replicas=2,
            cpu_limit="500m",
            memory_limit="512Mi",
            auto_scale=True,
            health_check_enabled=True,
        )

        # Deploy model
        deployment_id = await deployment.deploy_model(
            "deployment_model", SemanticVersion("1.0.0"), config
        )
        assert deployment_id is not None

        # Check deployment status
        status = await deployment.get_deployment_status(deployment_id)
        assert status["status"] in ["deploying", "running"]
        assert status["environment"] == "testing"

        # List deployments
        deployments = await deployment.list_deployments()
        assert len(deployments) == 1
        assert deployments[0]["model_name"] == "deployment_model"

        # Update deployment
        updated_config = DeploymentConfig(
            environment="testing",
            replicas=3,  # Scale up
            cpu_limit="500m",
            memory_limit="512Mi",
            auto_scale=True,
            health_check_enabled=True,
        )

        updated = await deployment.update_deployment(deployment_id, updated_config)
        assert updated

        # Verify update
        updated_status = await deployment.get_deployment_status(deployment_id)
        assert updated_status["config"]["replicas"] == 3

        # Stop deployment
        stopped = await deployment.stop_deployment(deployment_id)
        assert stopped

        # Verify stopped
        final_status = await deployment.get_deployment_status(deployment_id)
        assert final_status["status"] == "stopped"


@pytest.mark.asyncio
class TestModelMonitoringIntegration:
    """Integration tests for model monitoring."""

    async def test_model_monitoring_lifecycle(
        self, sample_model, sample_dataset, temp_model_dir
    ):
        """Test complete model monitoring lifecycle."""
        registry = ModelRegistry(storage_path=temp_model_dir)
        monitoring = ModelMonitoring(registry)

        # Register model
        metadata = await registry.register_model(
            sample_model, name="monitored_model", version=SemanticVersion("1.0.0")
        )

        # Start monitoring
        monitor_id = await monitoring.start_monitoring(
            "monitored_model", SemanticVersion("1.0.0"), sample_dataset
        )
        assert monitor_id is not None

        # Record performance metrics
        metrics = PerformanceMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            execution_time_ms=45.0,
            memory_usage_mb=128.0,
            throughput_per_second=1000.0,
        )

        await monitoring.record_metrics(monitor_id, metrics)

        # Get monitoring status
        status = await monitoring.get_monitoring_status(monitor_id)
        assert status["status"] == "active"
        assert status["metrics_count"] > 0

        # Check for drift detection
        # Simulate data drift by creating different dataset
        drift_data = pd.DataFrame(
            {
                "feature_1": np.random.randn(100) + 2,  # Shifted distribution
                "feature_2": np.random.randn(100) * 2,  # Different variance
                "feature_3": np.random.randn(100),
                "timestamp": pd.date_range(start="2023-02-01", periods=100, freq="H"),
            }
        )

        drift_dataset = Dataset(
            name="drift_dataset",
            data=drift_data,
            description="Dataset with potential drift",
        )

        drift_detected = await monitoring.detect_drift(monitor_id, drift_dataset)
        assert isinstance(drift_detected, dict)
        assert "drift_score" in drift_detected

        # Generate monitoring report
        report = await monitoring.generate_monitoring_report(monitor_id)
        assert "model_name" in report
        assert "monitoring_period" in report
        assert "performance_summary" in report
        assert "drift_analysis" in report

        # Stop monitoring
        stopped = await monitoring.stop_monitoring(monitor_id)
        assert stopped

        # Verify stopped
        final_status = await monitoring.get_monitoring_status(monitor_id)
        assert final_status["status"] == "stopped"

    async def test_drift_detection_integration(
        self, sample_model, sample_dataset, temp_model_dir
    ):
        """Test drift detection integration."""
        registry = ModelRegistry(storage_path=temp_model_dir)
        monitoring = ModelMonitoring(registry)

        # Register and start monitoring
        await registry.register_model(
            sample_model, name="drift_model", version=SemanticVersion("1.0.0")
        )

        monitor_id = await monitoring.start_monitoring(
            "drift_model", SemanticVersion("1.0.0"), sample_dataset
        )

        # Test different types of drift

        # 1. Feature drift (distribution change)
        feature_drift_data = pd.DataFrame(
            {
                "feature_1": np.random.exponential(2, 100),  # Different distribution
                "feature_2": np.random.randn(100),
                "feature_3": np.random.randn(100),
                "timestamp": pd.date_range(start="2023-02-01", periods=100, freq="H"),
            }
        )

        feature_drift_dataset = Dataset(
            name="feature_drift_dataset",
            data=feature_drift_data,
            description="Dataset with feature drift",
        )

        feature_drift_result = await monitoring.detect_drift(
            monitor_id, feature_drift_dataset
        )
        assert feature_drift_result["drift_detected"]
        assert "feature_drift" in feature_drift_result

        # 2. Concept drift (relationship change)
        concept_drift_data = sample_dataset.data.copy()
        # Simulate concept drift by adding noise correlation
        concept_drift_data["feature_1"] = (
            concept_drift_data["feature_1"] + 0.5 * concept_drift_data["feature_2"]
        )

        concept_drift_dataset = Dataset(
            name="concept_drift_dataset",
            data=concept_drift_data,
            description="Dataset with concept drift",
        )

        concept_drift_result = await monitoring.detect_drift(
            monitor_id, concept_drift_dataset
        )
        assert isinstance(concept_drift_result, dict)

        # 3. No drift (similar data)
        no_drift_data = sample_dataset.data + np.random.normal(
            0, 0.01, sample_dataset.data.shape
        )  # Minimal noise
        no_drift_data["timestamp"] = pd.date_range(
            start="2023-03-01", periods=100, freq="H"
        )

        no_drift_dataset = Dataset(
            name="no_drift_dataset",
            data=no_drift_data,
            description="Dataset with no significant drift",
        )

        no_drift_result = await monitoring.detect_drift(monitor_id, no_drift_dataset)
        assert no_drift_result["drift_score"] < 0.5  # Low drift score


@pytest.mark.asyncio
class TestAutoMLPipelineIntegration:
    """Integration tests for AutoML pipeline."""

    async def test_automl_pipeline_lifecycle(self, sample_dataset, temp_model_dir):
        """Test complete AutoML pipeline lifecycle."""
        registry = ModelRegistry(storage_path=temp_model_dir)
        automl = AutoMLPipeline(registry)

        # Configure AutoML
        config = {
            "algorithms": ["isolation_forest", "one_class_svm", "local_outlier_factor"],
            "parameter_ranges": {
                "isolation_forest": {
                    "contamination": [0.05, 0.1, 0.15],
                    "n_estimators": [50, 100, 200],
                },
                "one_class_svm": {"gamma": ["scale", "auto"], "nu": [0.05, 0.1, 0.15]},
            },
            "evaluation_metrics": ["precision", "recall", "f1_score"],
            "optimization_target": "f1_score",
            "max_trials": 5,
            "timeout_minutes": 10,
        }

        # Run AutoML pipeline
        pipeline_id = await automl.run_pipeline(
            sample_dataset, config, experiment_name="test_automl_experiment"
        )
        assert pipeline_id is not None

        # Monitor pipeline progress
        progress = await automl.get_pipeline_progress(pipeline_id)
        assert "status" in progress
        assert "trials_completed" in progress
        assert "best_score" in progress

        # Wait for completion (with timeout)
        max_wait_time = 30  # seconds
        wait_time = 0
        while wait_time < max_wait_time:
            status = await automl.get_pipeline_status(pipeline_id)
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(1)
            wait_time += 1

        # Get pipeline results
        results = await automl.get_pipeline_results(pipeline_id)
        assert "best_model" in results
        assert "all_trials" in results
        assert "experiment_summary" in results

        # Verify best model
        best_model_info = results["best_model"]
        assert "model_name" in best_model_info
        assert "score" in best_model_info
        assert "parameters" in best_model_info

        # Get experiment history
        experiments = await automl.list_experiments()
        assert len(experiments) >= 1
        experiment_found = any(
            exp["name"] == "test_automl_experiment" for exp in experiments
        )
        assert experiment_found

        # Get specific experiment details
        experiment_details = await automl.get_experiment_details(
            "test_automl_experiment"
        )
        assert experiment_details["name"] == "test_automl_experiment"
        assert "models_tested" in experiment_details
        assert "best_performance" in experiment_details

    async def test_model_comparison_integration(self, sample_dataset, temp_model_dir):
        """Test model comparison integration."""
        registry = ModelRegistry(storage_path=temp_model_dir)
        automl = AutoMLPipeline(registry)

        # Create multiple models for comparison
        models_to_compare = []

        # Model 1: Isolation Forest
        model1 = Model(
            name="isolation_forest_model",
            algorithm="isolation_forest",
            parameters={"contamination": 0.1, "n_estimators": 100},
            description="Isolation Forest model",
        )

        metadata1 = await registry.register_model(
            model1, name="iso_forest_v1", version=SemanticVersion("1.0.0")
        )
        models_to_compare.append(("iso_forest_v1", SemanticVersion("1.0.0")))

        # Model 2: One-Class SVM
        model2 = Model(
            name="one_class_svm_model",
            algorithm="one_class_svm",
            parameters={"gamma": "scale", "nu": 0.1},
            description="One-Class SVM model",
        )

        metadata2 = await registry.register_model(
            model2, name="svm_v1", version=SemanticVersion("1.0.0")
        )
        models_to_compare.append(("svm_v1", SemanticVersion("1.0.0")))

        # Compare models
        comparison_result = await automl.compare_models(
            models_to_compare,
            sample_dataset,
            metrics=["accuracy", "precision", "recall", "f1_score"],
        )

        # Verify comparison results
        assert "comparison_summary" in comparison_result
        assert "model_performances" in comparison_result
        assert "recommendations" in comparison_result

        # Check that all models were evaluated
        performances = comparison_result["model_performances"]
        assert len(performances) == 2

        model_names = [perf["model_name"] for perf in performances]
        assert "iso_forest_v1" in model_names
        assert "svm_v1" in model_names

        # Verify performance metrics
        for performance in performances:
            assert "metrics" in performance
            assert "execution_time_ms" in performance
            metrics = performance["metrics"]
            assert all(
                metric in metrics
                for metric in ["accuracy", "precision", "recall", "f1_score"]
            )


@pytest.mark.asyncio
class TestModelManagementEndToEnd:
    """End-to-end integration tests for model management."""

    async def test_complete_model_lifecycle(
        self, sample_dataset, sample_detector, temp_model_dir
    ):
        """Test complete model lifecycle from training to deployment."""
        registry = ModelRegistry(storage_path=temp_model_dir)
        versioning = ModelVersioning(registry)
        deployment = ModelDeployment(registry)
        monitoring = ModelMonitoring(registry)

        # 1. Train and register model
        sample_detector.fit(sample_dataset)

        model = Model(
            name="end_to_end_model",
            algorithm=sample_detector.algorithm,
            parameters=sample_detector.model.get_params(),
            description="End-to-end test model",
        )

        metadata = await registry.register_model(
            model, name="e2e_model", version=SemanticVersion("1.0.0")
        )
        assert metadata.status == ModelStatus.REGISTERED

        # 2. Validate model
        await registry.update_model_metadata(
            "e2e_model", SemanticVersion("1.0.0"), status=ModelStatus.VALIDATED
        )

        # 3. Deploy model
        config = DeploymentConfig(
            environment="production",
            replicas=1,
            cpu_limit="500m",
            memory_limit="512Mi",
            auto_scale=False,
            health_check_enabled=True,
        )

        deployment_id = await deployment.deploy_model(
            "e2e_model", SemanticVersion("1.0.0"), config
        )

        # 4. Start monitoring
        monitor_id = await monitoring.start_monitoring(
            "e2e_model", SemanticVersion("1.0.0"), sample_dataset
        )

        # 5. Record performance metrics
        metrics = PerformanceMetrics(
            accuracy=0.92,
            precision=0.89,
            recall=0.85,
            f1_score=0.87,
            execution_time_ms=55.0,
            memory_usage_mb=256.0,
            throughput_per_second=800.0,
        )

        await monitoring.record_metrics(monitor_id, metrics)

        # 6. Create new version with improvements
        improved_model = Model(
            name="end_to_end_model",
            algorithm=sample_detector.algorithm,
            parameters={
                **sample_detector.model.get_params(),
                "param1": 2.0,
            },  # Improved parameter
            description="Improved end-to-end test model",
        )

        v2_metadata = await versioning.create_version(
            improved_model,
            name="e2e_model",
            version=SemanticVersion("1.1.0"),
            description="Performance improvements",
        )

        # 7. Deploy new version (rolling update)
        new_deployment_id = await deployment.deploy_model(
            "e2e_model", SemanticVersion("1.1.0"), config
        )

        # 8. Monitor both versions
        new_monitor_id = await monitoring.start_monitoring(
            "e2e_model", SemanticVersion("1.1.0"), sample_dataset
        )

        # 9. Compare performance between versions
        comparison = await versioning.compare_versions(
            "e2e_model", SemanticVersion("1.0.0"), SemanticVersion("1.1.0")
        )

        assert "parameter_changes" in comparison
        assert comparison["parameter_changes"]["param1"]["old"] == 1.0
        assert comparison["parameter_changes"]["param1"]["new"] == 2.0

        # 10. Generate comprehensive report
        monitoring_report = await monitoring.generate_monitoring_report(monitor_id)
        assert monitoring_report["model_name"] == "e2e_model"

        # 11. Clean up (stop monitoring and deployments)
        await monitoring.stop_monitoring(monitor_id)
        await monitoring.stop_monitoring(new_monitor_id)
        await deployment.stop_deployment(deployment_id)
        await deployment.stop_deployment(new_deployment_id)

        # Verify final state
        final_status = await deployment.get_deployment_status(deployment_id)
        assert final_status["status"] == "stopped"

    async def test_global_model_registry_integration(self, sample_model):
        """Test global model registry integration."""
        # Test global registry retrieval
        registry1 = get_model_registry()
        registry2 = get_model_registry()

        # Verify singleton behavior
        assert registry1 is registry2
        assert isinstance(registry1, ModelRegistry)

        # Test global registry functionality
        metadata = await registry1.register_model(
            sample_model, name="global_test_model", version=SemanticVersion("1.0.0")
        )

        # Verify functionality
        assert metadata.name == "global_test_model"
        assert metadata.version.major == 1

        # Verify persistence across references
        models = await registry2.list_models()
        assert len(models) >= 1
        assert any(m.name == "global_test_model" for m in models)


@pytest.mark.asyncio
class TestModelManagementErrorHandling:
    """Test error handling in model management."""

    async def test_model_registry_error_handling(self, sample_model):
        """Test model registry error handling."""
        registry = ModelRegistry(storage_path="/invalid/path")

        # Test registration with invalid path
        with pytest.raises(Exception):
            await registry.register_model(
                sample_model, name="error_test_model", version=SemanticVersion("1.0.0")
            )

        # Test getting non-existent model
        registry = ModelRegistry()  # Use default path
        result = await registry.get_model("nonexistent_model", SemanticVersion("1.0.0"))
        assert result is None

        # Test invalid version format
        with pytest.raises(Exception):
            await registry.register_model(
                sample_model,
                name="invalid_version_model",
                version="invalid.version",  # Should be SemanticVersion
            )

    async def test_deployment_error_handling(self, temp_model_dir):
        """Test deployment error handling."""
        registry = ModelRegistry(storage_path=temp_model_dir)
        deployment = ModelDeployment(registry)

        # Test deploying non-existent model
        config = DeploymentConfig(environment="test")

        deployment_id = await deployment.deploy_model(
            "nonexistent_model", SemanticVersion("1.0.0"), config
        )

        # Should handle gracefully
        assert deployment_id is None or isinstance(deployment_id, str)

        # Test invalid deployment config
        invalid_config = DeploymentConfig(
            environment="",  # Empty environment
            replicas=-1,  # Invalid replica count
        )

        # Should handle validation errors
        with pytest.raises(Exception):
            await deployment.deploy_model(
                "test_model", SemanticVersion("1.0.0"), invalid_config
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
