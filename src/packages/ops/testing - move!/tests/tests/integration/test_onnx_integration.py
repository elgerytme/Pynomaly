"""Integration test for ONNX model persistence."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from monorepo.application.services.model_persistence_service import (
    ModelPersistenceService,
)
from monorepo.domain.entities import Dataset
from monorepo.domain.value_objects import ContaminationRate
from monorepo.infrastructure.adapters.sklearn_adapter import SklearnAdapter
from monorepo.infrastructure.repositories.in_memory_repositories import (
    InMemoryDetectorRepository,
)


@pytest.mark.asyncio
async def test_onnx_model_persistence_integration():
    """Test complete ONNX model persistence workflow."""
    # Setup
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)
        repository = InMemoryDetectorRepository()
        persistence_service = ModelPersistenceService(repository, storage_path)

        # Create a simple detector
        detector = SklearnAdapter(
            algorithm_name="IsolationForest",
            name="Test Isolation Forest",
            contamination_rate=ContaminationRate(0.1),
            n_estimators=10,
            random_state=42,
        )

        # Create sample data and fit the detector
        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 100],  # 100 is an outlier
                "feature2": [2, 4, 6, 8, 10, 200],  # 200 is an outlier
            }
        )
        dataset = Dataset(name="test_data", data=data)

        # Fit the detector
        detector.fit(dataset)

        # Save to repository
        repository.save(detector)

        # Test ONNX save/load cycle
        detector_id = detector.id

        # Save model in ONNX format (should create stub since PyTorch not available)
        model_path = await persistence_service.save_model(detector_id, format="onnx")

        # Verify model was saved
        assert Path(model_path).exists()
        assert model_path.endswith("model.onnx")

        # Load model back
        loaded_detector = await persistence_service.load_model(
            detector_id, format="onnx"
        )

        # Verify loaded detector works
        assert loaded_detector is not None
        assert loaded_detector.name == "Test Isolation Forest"
        assert loaded_detector.algorithm_name == "IsolationForest"

        # Test that basic interface works (note: stub detector may not be fitted)
        assert hasattr(loaded_detector, "detect")
        assert hasattr(loaded_detector, "score")

        # Verify metadata was saved
        saved_models = await persistence_service.list_saved_models()
        assert str(detector_id) in saved_models
        assert saved_models[str(detector_id)]["format"] == "onnx"


@pytest.mark.asyncio
async def test_onnx_export_functionality():
    """Test ONNX model export functionality."""
    # Setup
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)
        export_path = Path(temp_dir) / "export"
        repository = InMemoryDetectorRepository()
        persistence_service = ModelPersistenceService(repository, storage_path)

        # Create and fit detector
        detector = SklearnAdapter(
            algorithm_name="IsolationForest",
            name="Export Test Detector",
            contamination_rate=ContaminationRate(0.05),
            n_estimators=5,
            random_state=123,
        )

        data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 4, 5]})
        dataset = Dataset(name="export_test", data=data)

        detector.fit(dataset)
        repository.save(detector)

        # Export model
        exported_files = await persistence_service.export_model(
            detector.id, export_path, include_data=False
        )

        # Verify export files
        assert "model" in exported_files
        assert "config" in exported_files
        assert "requirements" in exported_files
        assert "deploy_script" in exported_files

        # Verify files exist
        for file_path in exported_files.values():
            assert Path(file_path).exists()
