#!/usr/bin/env python3
"""Integration tests for MLOps integration services.

These tests verify that the unified model registry and experiment tracking
integration work correctly with the anomaly detection system.
"""

import pytest
import numpy as np
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from ai.mlops.domain.services.mlops_service import MLOpsService
from ai.mlops.infrastructure.repositories.model_repository import ModelRepository
from anomaly_detection.application.services.mlops import (
    UnifiedModelRegistry,
    ExperimentTrackingIntegration,
    ModelRegistrationRequest,
    initialize_unified_model_registry,
    initialize_experiment_tracking_integration,
)


class TestMLOpsIntegration:
    """Test MLOps integration functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def anomaly_mlops_service(self):
        """Create anomaly detection MLOps service."""
        return MLOpsService(
            model_repository=None,  # Using in-memory for tests
            tracking_backend="local"
        )
    
    @pytest.fixture
    def model_repository(self, temp_dir):
        """Create model repository."""
        return ModelRepository(storage_path=Path(temp_dir))
    
    @pytest.fixture
    def unified_registry(self, anomaly_mlops_service, model_repository):
        """Create unified model registry."""
        return UnifiedModelRegistry(
            anomaly_mlops_service=anomaly_mlops_service,
            anomaly_model_repository=model_repository,
            mlops_model_service=None  # No MLOps service for basic tests
        )
    
    @pytest.fixture
    def experiment_integration(self, anomaly_mlops_service):
        """Create experiment tracking integration."""
        return ExperimentTrackingIntegration(
            anomaly_mlops_service=anomaly_mlops_service,
            mlops_experiment_service=None  # No MLOps service for basic tests
        )
    
    @pytest.fixture
    def sample_model(self):
        """Create sample model for testing."""
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(contamination=0.1, random_state=42)
        
        # Fit with dummy data
        X = np.random.randn(100, 5)
        model.fit(X)
        
        return model
    
    @pytest.fixture
    def sample_registration_request(self, sample_model):
        """Create sample model registration request."""
        return ModelRegistrationRequest(
            name="test_anomaly_model",
            description="Test anomaly detection model",
            algorithm="isolation_forest",
            model_object=sample_model,
            performance_metrics={
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.87,
                "f1_score": 0.88,
                "training_time": 5.2,
                "inference_time": 0.001
            },
            training_data_info={
                "samples": 1000,
                "features": 5,
                "anomaly_rate": 0.1
            },
            deployment_config={
                "serving_type": "batch",
                "batch_size": 100
            },
            tags=["anomaly_detection", "isolation_forest", "test"],
            created_by="test_user",
            framework="scikit-learn",
            use_cases=["fraud_detection", "outlier_analysis"],
            data_requirements={"min_features": 5, "data_format": "numpy"}
        )
    
    async def test_unified_model_registry_initialization(self, unified_registry):
        """Test unified model registry initialization."""
        assert unified_registry is not None
        assert not unified_registry.mlops_integration_enabled
        assert unified_registry.anomaly_mlops_service is not None
        assert unified_registry.anomaly_model_repository is not None
        
        # Check registry stats
        stats = unified_registry.get_registry_stats()
        assert stats["total_models"] == 0
        assert stats["anomaly_detection_integrated"] == 0
        assert stats["mlops_integrated"] == 0
        assert not stats["mlops_integration_enabled"]
    
    async def test_model_registration(self, unified_registry, sample_registration_request):
        """Test model registration in unified registry."""
        # Register model
        metadata = await unified_registry.register_model(sample_registration_request)
        
        # Verify metadata
        assert metadata is not None
        assert metadata.name == "test_anomaly_model"
        assert metadata.algorithm == "isolation_forest"
        assert metadata.framework == "scikit-learn"
        assert metadata.performance_metrics["accuracy"] == 0.92
        assert metadata.anomaly_model_id is not None
        assert metadata.mlops_model_id is None  # No MLOps integration
        
        # Verify registry has the model
        stats = unified_registry.get_registry_stats()
        assert stats["total_models"] == 1
        assert stats["anomaly_detection_integrated"] == 1
        
        # Retrieve model
        retrieved_model = await unified_registry.get_model(metadata.model_id)
        assert retrieved_model is not None
        assert retrieved_model.name == metadata.name
        assert retrieved_model.model_id == metadata.model_id
    
    async def test_model_listing_and_filtering(self, unified_registry, sample_registration_request):
        """Test model listing and filtering."""
        # Register multiple models
        model1 = await unified_registry.register_model(sample_registration_request)
        
        # Create second model request
        request2 = ModelRegistrationRequest(
            name="test_anomaly_model_2",
            description="Second test model",
            algorithm="local_outlier_factor",
            model_object=sample_registration_request.model_object,
            performance_metrics={"accuracy": 0.85, "precision": 0.82},
            training_data_info={"samples": 500},
            tags=["anomaly_detection", "lof"],
            created_by="test_user"
        )
        model2 = await unified_registry.register_model(request2)
        
        # List all models
        all_models = await unified_registry.list_models()
        assert len(all_models) == 2
        
        # Filter by algorithm
        isolation_models = await unified_registry.list_models({"algorithm": "isolation_forest"})
        assert len(isolation_models) == 1
        assert isolation_models[0].name == "test_anomaly_model"
        
        # Filter by tags
        lof_models = await unified_registry.list_models({"tags": ["lof"]})
        assert len(lof_models) == 1
        assert lof_models[0].name == "test_anomaly_model_2"
        
        # Filter by created_by
        user_models = await unified_registry.list_models({"created_by": "test_user"})
        assert len(user_models) == 2
    
    async def test_model_promotion(self, unified_registry, sample_registration_request):
        """Test model promotion workflow."""
        # Register model
        metadata = await unified_registry.register_model(sample_registration_request)
        
        # Promote to production
        promoted_model = await unified_registry.promote_model(
            metadata.model_id, "production", "test_user"
        )
        
        assert promoted_model is not None
        assert promoted_model.model_id == metadata.model_id
        
        # Verify promotion in anomaly detection system
        versions = unified_registry.anomaly_mlops_service.get_model_versions(metadata.name)
        assert len(versions) > 0
        production_version = next((v for v in versions if v.status == "production"), None)
        assert production_version is not None
    
    async def test_experiment_tracking_integration_initialization(self, experiment_integration):
        """Test experiment tracking integration initialization."""
        assert experiment_integration is not None
        assert not experiment_integration.mlops_integration_enabled
        assert experiment_integration.anomaly_mlops_service is not None
        
        # Check integration stats
        stats = experiment_integration.get_integration_stats()
        assert stats["total_experiments"] == 0
        assert stats["total_runs"] == 0
        assert not stats["mlops_integration_enabled"]
    
    async def test_unified_experiment_creation(self, experiment_integration):
        """Test unified experiment creation."""
        # Create experiment
        experiment = await experiment_integration.create_experiment(
            name="test_experiment",
            description="Test experiment for anomaly detection",
            tags=["test", "anomaly_detection"],
            parameters={"contamination": 0.1, "n_estimators": 100},
            created_by="test_user"
        )
        
        # Verify experiment
        assert experiment is not None
        assert experiment.name == "test_experiment"
        assert experiment.created_by == "test_user"
        assert "test" in experiment.tags
        assert experiment.anomaly_experiment_id is not None
        assert experiment.mlops_experiment_id is None  # No MLOps integration
        
        # Verify in integration stats
        stats = experiment_integration.get_integration_stats()
        assert stats["total_experiments"] == 1
        
        # Retrieve experiment
        retrieved_exp = await experiment_integration.get_experiment(experiment.experiment_id)
        assert retrieved_exp is not None
        assert retrieved_exp.name == experiment.name
    
    async def test_unified_experiment_run_workflow(self, experiment_integration, sample_model):
        """Test complete experiment run workflow."""
        # Create experiment
        experiment = await experiment_integration.create_experiment(
            name="run_test_experiment",
            description="Test experiment runs",
            created_by="test_user"
        )
        
        # Start run
        run = await experiment_integration.start_run(
            experiment_id=experiment.experiment_id,
            run_name="test_run_1",
            parameters={"contamination": 0.1, "random_state": 42},
            tags=["test_run"],
            created_by="test_user"
        )
        
        # Verify run started
        assert run is not None
        assert run.experiment_id == experiment.experiment_id
        assert run.name == "test_run_1"
        assert run.parameters["contamination"] == 0.1
        assert run.anomaly_run_id is not None
        
        # Log metrics
        experiment_integration.log_metrics(
            run.run_id,
            {"accuracy": 0.93, "precision": 0.91, "recall": 0.89}
        )
        
        # Log parameters
        experiment_integration.log_parameters(
            run.run_id,
            {"algorithm": "isolation_forest", "training_samples": 1000}
        )
        
        # Log model
        model_path = experiment_integration.log_model(
            run.run_id,
            sample_model,
            "test_model"
        )
        assert model_path is not None
        
        # End run
        completed_run = await experiment_integration.end_run(run.run_id, "completed")
        assert completed_run.ended_at is not None
        assert completed_run.duration_seconds is not None
        
        # Verify run in stats
        stats = experiment_integration.get_integration_stats()
        assert stats["total_runs"] == 1
        
        # List runs for experiment
        runs = await experiment_integration.list_runs(experiment.experiment_id)
        assert len(runs) == 1
        assert runs[0].run_id == run.run_id
    
    async def test_experiment_run_comparison(self, experiment_integration, sample_model):
        """Test experiment run comparison functionality."""
        # Create experiment
        experiment = await experiment_integration.create_experiment(
            name="comparison_test",
            created_by="test_user"
        )
        
        # Create multiple runs with different performance
        run_ids = []
        
        for i in range(3):
            run = await experiment_integration.start_run(
                experiment_id=experiment.experiment_id,
                run_name=f"run_{i}",
                parameters={"contamination": 0.1 + i * 0.05},
                created_by="test_user"
            )
            
            # Log different performance metrics
            metrics = {
                "accuracy": 0.9 - i * 0.02,
                "precision": 0.88 - i * 0.01,
                "f1_score": 0.85 - i * 0.015
            }
            experiment_integration.log_metrics(run.run_id, metrics)
            
            await experiment_integration.end_run(run.run_id, "completed")
            run_ids.append(run.run_id)
        
        # Compare runs
        comparison = await experiment_integration.compare_runs(
            run_ids=run_ids,
            metrics_to_compare=["accuracy", "precision", "f1_score"]
        )
        
        # Verify comparison
        assert comparison is not None
        assert len(comparison.run_ids) == 3
        assert len(comparison.performance_summary) == 3
        assert comparison.best_run is not None
        assert len(comparison.recommendations) > 0
        
        # Best run should be the first one (highest metrics)
        best_run_metrics = comparison.performance_summary[run_ids[0]]
        assert best_run_metrics["accuracy"] == 0.9
    
    async def test_health_checks(self, unified_registry, experiment_integration):
        """Test health check functionality."""
        # Test unified registry health check
        registry_health = unified_registry.health_check()
        assert registry_health["unified_registry"] == "healthy"
        assert registry_health["anomaly_detection_service"] == "healthy"
        assert registry_health["mlops_service"] == "disabled"
        
        # Test experiment integration health check
        integration_health = experiment_integration.health_check()
        assert integration_health["experiment_tracking_integration"] == "healthy"
        assert integration_health["anomaly_detection_service"] == "healthy"
        assert integration_health["mlops_service"] == "disabled"
    
    async def test_global_initialization_functions(self, anomaly_mlops_service, model_repository):
        """Test global initialization functions."""
        # Test unified model registry initialization
        registry = initialize_unified_model_registry(
            anomaly_mlops_service=anomaly_mlops_service,
            anomaly_model_repository=model_repository
        )
        assert registry is not None
        
        # Test experiment tracking integration initialization
        integration = initialize_experiment_tracking_integration(
            anomaly_mlops_service=anomaly_mlops_service
        )
        assert integration is not None
    
    async def test_error_handling(self, unified_registry, experiment_integration):
        """Test error handling in integration services."""
        # Test getting non-existent model
        model = await unified_registry.get_model("non_existent_id")
        assert model is None
        
        # Test getting non-existent experiment
        experiment = await experiment_integration.get_experiment("non_existent_id")
        assert experiment is None
        
        # Test starting run for non-existent experiment
        with pytest.raises(ValueError, match="Experiment .* not found"):
            await experiment_integration.start_run(
                experiment_id="non_existent_id",
                run_name="test_run"
            )
        
        # Test logging metrics for non-existent run
        with pytest.raises(ValueError, match="Run .* not found"):
            experiment_integration.log_metrics("non_existent_run", {"accuracy": 0.9})
    
    def test_data_serialization(self, sample_registration_request):
        """Test data serialization and deserialization."""
        # Test UnifiedModelMetadata serialization
        from anomaly_detection.application.services.mlops import UnifiedModelMetadata
        
        metadata = UnifiedModelMetadata(
            model_id="test_id",
            name="test_model",
            description="test description",
            algorithm="test_algorithm",
            framework="scikit-learn",
            version="1.0.0",
            performance_metrics={"accuracy": 0.9},
            training_data_info={"samples": 1000},
            deployment_config={"batch_size": 100},
            tags=["test"],
            created_by="test_user",
            created_at=datetime.now()
        )
        
        # Test to_dict conversion
        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["model_id"] == "test_id"
        assert metadata_dict["name"] == "test_model"
        assert metadata_dict["performance_metrics"]["accuracy"] == 0.9
        
        # Test ModelRegistrationRequest (already has to_dict through dataclass functionality)
        assert hasattr(sample_registration_request, 'name')
        assert hasattr(sample_registration_request, 'algorithm')
        assert hasattr(sample_registration_request, 'performance_metrics')


if __name__ == "__main__":
    print("MLOps Integration Test Suite")
    print("=" * 40)
    
    # Quick smoke test
    try:
        from anomaly_detection.application.services.mlops import UnifiedModelRegistry
        from ai.mlops.domain.services.mlops_service import MLOpsService
        from ai.mlops.infrastructure.repositories.model_repository import ModelRepository
        
        print("✓ All imports successful")
        
        # Test basic initialization
        mlops_service = MLOpsService(None, "local")
        model_repo = ModelRepository()
        registry = UnifiedModelRegistry(mlops_service, model_repo)
        
        print("✓ Basic initialization successful")
        print("✓ MLOps integration tests ready to run")
        
    except Exception as e:
        print(f"✗ Smoke test failed: {e}")
        print("Some tests may not run properly")