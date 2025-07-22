"""Integration tests for ModelRepository."""

import pytest
import json
import pickle
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository
from anomaly_detection.domain.entities.model import Model, ModelMetadata, ModelStatus, SerializationFormat
from anomaly_detection.domain.services.detection_service import DetectionService
from .conftest import assert_model_saved_correctly, create_test_model_metadata


class TestModelRepositoryIntegration:
    """Integration tests for ModelRepository with real file operations."""
    
    def test_repository_initialization(self, temp_dir: Path):
        """Test repository initialization creates necessary directories."""
        model_repo = ModelRepository(str(temp_dir / "models"))
        
        # Directory should be created
        assert (temp_dir / "models").exists()
        assert (temp_dir / "models").is_dir()
    
    def test_save_and_load_complete_workflow(self, model_repository: ModelRepository, 
                                           detection_service: DetectionService, 
                                           test_data: Dict[str, Any]):
        """Test complete save and load workflow with real trained model."""
        # Train a real model
        train_data = test_data['normal_data']
        detection_service.fit(train_data, algorithm="iforest", random_state=42)
        trained_model = detection_service._fitted_models["iforest"]
        
        # Create model entity
        metadata = ModelMetadata(
            model_id="test-iforest-001",
            name="Test Isolation Forest",
            algorithm="isolation_forest",
            status=ModelStatus.TRAINED,
            training_samples=train_data.shape[0],
            training_features=train_data.shape[1],
            contamination_rate=0.1,
            feature_names=["feature_1", "feature_2"],
            description="Integration test model"
        )
        
        model = Model(metadata=metadata, model_object=trained_model)
        
        # Save model
        saved_model_id = model_repository.save(model, SerializationFormat.PICKLE)
        assert saved_model_id == "test-iforest-001"
        
        # Verify files created
        assert_model_saved_correctly(Path(model_repository.storage_path), saved_model_id)
        
        # Load model
        loaded_model = model_repository.load(saved_model_id)
        
        # Verify model loaded correctly
        assert loaded_model.metadata.model_id == saved_model_id
        assert loaded_model.metadata.name == "Test Isolation Forest"
        assert loaded_model.metadata.algorithm == "isolation_forest"
        assert loaded_model.model_object is not None
        
        # Test that loaded model can make predictions
        test_data_array = test_data['data_only'][:10]  # Use subset for quick test
        
        # Original model predictions
        original_predictions = trained_model.predict(test_data_array)
        
        # Loaded model predictions
        loaded_predictions = loaded_model.model_object.predict(test_data_array)
        
        # Should be identical
        assert (original_predictions == loaded_predictions).all()
    
    def test_different_serialization_formats(self, model_repository: ModelRepository,
                                           detection_service: DetectionService,
                                           test_data: Dict[str, Any]):
        """Test saving and loading with different serialization formats."""
        # Train model
        train_data = test_data['normal_data']
        detection_service.fit(train_data, algorithm="iforest", random_state=42)
        trained_model = detection_service._fitted_models["iforest"]
        
        formats_to_test = [
            (SerializationFormat.PICKLE, "pkl"),
            (SerializationFormat.JOBLIB, "joblib")
        ]
        
        for format_enum, file_extension in formats_to_test:
            model_id = f"test-model-{format_enum.value}"
            
            metadata = ModelMetadata(
                model_id=model_id,
                name=f"Test Model {format_enum.value}",
                algorithm="isolation_forest",
                status=ModelStatus.TRAINED,
                training_samples=train_data.shape[0],
                training_features=train_data.shape[1]
            )
            
            model = Model(metadata=metadata, model_object=trained_model)
            
            # Save with specific format
            saved_id = model_repository.save(model, format_enum)
            assert saved_id == model_id
            
            # Check file extension
            model_dir = Path(model_repository.storage_path) / model_id
            model_file = model_dir / f"model.{file_extension}"
            assert model_file.exists()
            
            # Load and verify
            loaded_model = model_repository.load(model_id)
            assert loaded_model.metadata.model_id == model_id
            assert loaded_model.model_object is not None
            
            # Test prediction consistency
            test_sample = test_data['data_only'][:5]
            original_pred = trained_model.predict(test_sample)
            loaded_pred = loaded_model.model_object.predict(test_sample)
            assert (original_pred == loaded_pred).all()
    
    def test_model_metadata_persistence(self, model_repository: ModelRepository):
        """Test that model metadata is correctly persisted and loaded."""
        metadata = ModelMetadata(
            model_id="metadata-test-001",
            name="Metadata Test Model",
            algorithm="isolation_forest",
            version="2.1.0",
            status=ModelStatus.TRAINED,
            training_samples=500,
            training_features=10,
            contamination_rate=0.15,
            training_duration_seconds=45.7,
            accuracy=0.92,
            precision=0.88,
            recall=0.94,
            f1_score=0.91,
            feature_names=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
            hyperparameters={"n_estimators": 100, "contamination": 0.15, "random_state": 42},
            tags=["production", "v2", "high-accuracy"],
            description="Comprehensive metadata test model"
        )
        
        # Create a simple mock model object
        from sklearn.ensemble import IsolationForest
        mock_model = IsolationForest(random_state=42)
        
        model = Model(metadata=metadata, model_object=mock_model)
        
        # Save model
        model_id = model_repository.save(model)
        
        # Load and verify all metadata fields
        loaded_model = model_repository.load(model_id)
        loaded_metadata = loaded_model.metadata
        
        assert loaded_metadata.model_id == metadata.model_id
        assert loaded_metadata.name == metadata.name
        assert loaded_metadata.algorithm == metadata.algorithm
        assert loaded_metadata.version == metadata.version
        assert loaded_metadata.status == metadata.status
        assert loaded_metadata.training_samples == metadata.training_samples
        assert loaded_metadata.training_features == metadata.training_features
        assert loaded_metadata.contamination_rate == metadata.contamination_rate
        assert loaded_metadata.training_duration_seconds == metadata.training_duration_seconds
        assert loaded_metadata.accuracy == metadata.accuracy
        assert loaded_metadata.precision == metadata.precision
        assert loaded_metadata.recall == metadata.recall
        assert loaded_metadata.f1_score == metadata.f1_score
        assert loaded_metadata.feature_names == metadata.feature_names
        assert loaded_metadata.hyperparameters == metadata.hyperparameters
        assert loaded_metadata.tags == metadata.tags
        assert loaded_metadata.description == metadata.description
        assert loaded_metadata.created_at is not None
        assert loaded_metadata.updated_at is not None
    
    def test_model_listing_and_filtering(self, model_repository: ModelRepository):
        """Test model listing with filtering capabilities."""
        # Create multiple test models with different attributes
        models_data = [
            ("model-001", "isolation_forest", ModelStatus.TRAINED, ["prod", "v1"]),
            ("model-002", "local_outlier_factor", ModelStatus.TRAINED, ["test", "v1"]),
            ("model-003", "isolation_forest", ModelStatus.DEPLOYED, ["prod", "v2"]),
            ("model-004", "one_class_svm", ModelStatus.FAILED, ["test", "experimental"])
        ]
        
        from sklearn.ensemble import IsolationForest
        mock_model = IsolationForest(random_state=42)
        
        for model_id, algorithm, status, tags in models_data:
            metadata = ModelMetadata(
                model_id=model_id,
                name=f"Test {algorithm}",
                algorithm=algorithm,
                status=status,
                training_samples=100,
                training_features=5,
                tags=tags
            )
            model = Model(metadata=metadata, model_object=mock_model)
            model_repository.save(model)
        
        # Test listing all models
        all_models = model_repository.list_models()
        assert len(all_models) == 4
        
        # Test filtering by algorithm
        iforest_models = model_repository.list_models(algorithm="isolation_forest")
        assert len(iforest_models) == 2
        assert all(m["algorithm"] == "isolation_forest" for m in iforest_models)
        
        # Test filtering by status
        trained_models = model_repository.list_models(status=ModelStatus.TRAINED)
        assert len(trained_models) == 2
        assert all(m["status"] == "trained" for m in trained_models)
        
        deployed_models = model_repository.list_models(status=ModelStatus.DEPLOYED)
        assert len(deployed_models) == 1
        assert deployed_models[0]["model_id"] == "model-003"
    
    def test_model_deletion(self, model_repository: ModelRepository):
        """Test model deletion functionality."""
        # Create a test model
        metadata = ModelMetadata(
            model_id="deletion-test-001",
            name="Deletion Test Model",
            algorithm="isolation_forest",
            status=ModelStatus.TRAINED,
            training_samples=100,
            training_features=2
        )
        
        from sklearn.ensemble import IsolationForest
        mock_model = IsolationForest(random_state=42)
        model = Model(metadata=metadata, model_object=mock_model)
        
        # Save model
        model_id = model_repository.save(model)
        
        # Verify it exists
        model_dir = Path(model_repository.storage_path) / model_id
        assert model_dir.exists()
        assert (model_dir / "metadata.json").exists()
        
        # Delete model
        success = model_repository.delete(model_id)
        assert success is True
        
        # Verify it's gone
        assert not model_dir.exists()
        
        # Try to load deleted model
        with pytest.raises(FileNotFoundError):
            model_repository.load(model_id)
        
        # Try to delete non-existent model
        success = model_repository.delete("non-existent-model")
        assert success is False
    
    def test_repository_statistics(self, model_repository: ModelRepository):
        """Test repository statistics functionality."""
        # Initially empty
        stats = model_repository.get_repository_stats()
        assert stats["total_models"] == 0
        assert stats["storage_size_mb"] >= 0
        assert stats["by_status"] == {}
        assert stats["by_algorithm"] == {}
        
        # Add some models
        from sklearn.ensemble import IsolationForest
        mock_model = IsolationForest(random_state=42)
        
        models_to_create = [
            ("stats-001", "isolation_forest", ModelStatus.TRAINED),
            ("stats-002", "isolation_forest", ModelStatus.DEPLOYED),
            ("stats-003", "local_outlier_factor", ModelStatus.TRAINED)
        ]
        
        for model_id, algorithm, status in models_to_create:
            metadata = ModelMetadata(
                model_id=model_id,
                name=f"Stats test {model_id}",
                algorithm=algorithm,
                status=status,
                training_samples=100,
                training_features=2
            )
            model = Model(metadata=metadata, model_object=mock_model)
            model_repository.save(model)
        
        # Check updated stats
        stats = model_repository.get_repository_stats()
        assert stats["total_models"] == 3
        assert stats["storage_size_mb"] > 0
        
        # Check breakdown by status
        assert stats["by_status"]["trained"] == 2
        assert stats["by_status"]["deployed"] == 1
        
        # Check breakdown by algorithm
        assert stats["by_algorithm"]["isolation_forest"] == 2
        assert stats["by_algorithm"]["local_outlier_factor"] == 1
    
    def test_error_handling(self, model_repository: ModelRepository):
        """Test error handling for various edge cases."""
        # Try to load non-existent model
        with pytest.raises(FileNotFoundError):
            model_repository.load("non-existent-model")
        
        # Try to save model with invalid metadata
        with pytest.raises((ValueError, TypeError)):
            invalid_model = Model(metadata=None, model_object=None)
            model_repository.save(invalid_model)
    
    def test_concurrent_access(self, model_repository: ModelRepository):
        """Test basic concurrent access scenarios."""
        import threading
        from sklearn.ensemble import IsolationForest
        
        results = []
        errors = []
        
        def save_model(model_id: str):
            try:
                metadata = ModelMetadata(
                    model_id=f"concurrent-{model_id}",
                    name=f"Concurrent Model {model_id}",
                    algorithm="isolation_forest",
                    status=ModelStatus.TRAINED,
                    training_samples=100,
                    training_features=2
                )
                mock_model = IsolationForest(random_state=int(model_id))
                model = Model(metadata=metadata, model_object=mock_model)
                
                saved_id = model_repository.save(model)
                results.append(saved_id)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=save_model, args=[str(i)])
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert len(set(results)) == 5  # All unique
        
        # Verify all models were saved
        all_models = model_repository.list_models()
        concurrent_models = [m for m in all_models if m["model_id"].startswith("concurrent-")]
        assert len(concurrent_models) == 5
    
    def test_model_versioning_workflow(self, model_repository: ModelRepository):
        """Test model versioning and updates."""
        from sklearn.ensemble import IsolationForest
        
        # Create initial model
        metadata_v1 = ModelMetadata(
            model_id="versioned-model-001",
            name="Versioned Model",
            algorithm="isolation_forest",
            version="1.0.0",
            status=ModelStatus.TRAINED,
            training_samples=100,
            training_features=2,
            accuracy=0.85
        )
        
        model_v1 = Model(
            metadata=metadata_v1, 
            model_object=IsolationForest(random_state=42)
        )
        
        # Save v1
        model_repository.save(model_v1)
        
        # Create updated model (same ID, different version)
        metadata_v2 = ModelMetadata(
            model_id="versioned-model-001",
            name="Versioned Model",
            algorithm="isolation_forest",
            version="2.0.0",
            status=ModelStatus.DEPLOYED,
            training_samples=200,
            training_features=2,
            accuracy=0.92,
            description="Updated with more training data"
        )
        
        model_v2 = Model(
            metadata=metadata_v2,
            model_object=IsolationForest(n_estimators=200, random_state=42)
        )
        
        # Save v2 (should update existing)
        model_repository.save(model_v2)
        
        # Load and verify it's the updated version
        loaded_model = model_repository.load("versioned-model-001")
        assert loaded_model.metadata.version == "2.0.0"
        assert loaded_model.metadata.status == ModelStatus.DEPLOYED
        assert loaded_model.metadata.accuracy == 0.92
        assert loaded_model.metadata.training_samples == 200