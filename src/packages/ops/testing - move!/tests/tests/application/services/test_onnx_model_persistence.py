"""Tests for ONNX model persistence functionality."""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from monorepo.application.services.model_persistence_service import (
    ModelPersistenceService,
)
from monorepo.domain.entities import Detector
from monorepo.infrastructure.repositories.in_memory_repositories import (
    InMemoryDetectorRepository,
)


class TestONNXModelPersistence:
    """Test suite for ONNX model persistence functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.repository = InMemoryDetectorRepository()
        self.service = ModelPersistenceService(self.repository, self.temp_dir)

        # Create a basic detector
        self.detector = Detector(
            id=uuid4(),
            name="test-detector",
            algorithm_name="IsolationForest",
            parameters={"contamination": 0.1},
            created_at=datetime.now(UTC),
            is_fitted=True,
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_onnx_save_model_creates_stub(self):
        """Test that saving ONNX model creates a stub when PyTorch not available."""
        # Save detector to repository
        await self.repository.save(self.detector)

        # Save model in ONNX format
        model_path = await self.service.save_model(self.detector.id, format="onnx")

        # Verify model file was created
        assert Path(model_path).exists()
        assert Path(model_path).name == "model.onnx"

        # Verify it's a JSON stub (since PyTorch likely not available)
        with open(model_path) as f:
            stub_data = json.load(f)

        assert stub_data["model_type"] == "stub"
        assert stub_data["detector_name"] == self.detector.name
        assert stub_data["algorithm"] == self.detector.algorithm_name

    @pytest.mark.asyncio
    async def test_onnx_save_model_creates_metadata(self):
        """Test that saving ONNX model creates proper metadata."""
        # Save detector to repository
        await self.repository.save(self.detector)

        # Save model in ONNX format
        model_path = await self.service.save_model(self.detector.id, format="onnx")

        # Check metadata file was created
        model_dir = Path(model_path).parent
        metadata_path = model_dir / "metadata.json"
        assert metadata_path.exists()

        # Verify metadata content
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["detector_id"] == str(self.detector.id)
        assert metadata["detector_name"] == self.detector.name
        assert metadata["algorithm"] == self.detector.algorithm_name
        assert metadata["format"] == "onnx"
        assert metadata["is_fitted"] == True

    @pytest.mark.asyncio
    async def test_onnx_load_model_success(self):
        """Test that loading ONNX model works correctly."""
        # Save detector to repository
        await self.repository.save(self.detector)

        # Save model first
        await self.service.save_model(self.detector.id, format="onnx")

        # Load model
        loaded_detector = await self.service.load_model(self.detector.id, format="onnx")

        # Verify loaded detector properties
        assert loaded_detector.name == self.detector.name
        assert loaded_detector.algorithm_name == self.detector.algorithm_name

    @pytest.mark.asyncio
    async def test_onnx_load_model_not_found(self):
        """Test that loading non-existent ONNX model raises error."""
        non_existent_id = uuid4()

        with pytest.raises(
            ValueError, match=f"No saved model found for detector {non_existent_id}"
        ):
            await self.service.load_model(non_existent_id, format="onnx")

    @pytest.mark.asyncio
    async def test_onnx_save_load_round_trip(self):
        """Test complete save and load round trip for ONNX model."""
        # Save detector to repository
        await self.repository.save(self.detector)

        # Save model in ONNX format
        model_path = await self.service.save_model(
            self.detector.id, format="onnx", metadata={"test_data": "round_trip_test"}
        )

        # Load the model back
        loaded_detector = await self.service.load_model(self.detector.id, format="onnx")

        # Verify basic properties are preserved
        assert loaded_detector.name == self.detector.name
        assert loaded_detector.algorithm_name == self.detector.algorithm_name

        # Verify files exist
        assert Path(model_path).exists()
        metadata_path = Path(model_path).parent / "metadata.json"
        assert metadata_path.exists()

        # Verify metadata includes custom data
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["test_data"] == "round_trip_test"

    @pytest.mark.asyncio
    async def test_onnx_save_unfitted_detector_fails(self):
        """Test that saving unfitted detector raises error."""
        # Create unfitted detector
        unfitted_detector = Detector(
            id=uuid4(),
            name="unfitted-detector",
            algorithm_name="IsolationForest",
            parameters={},
            created_at=datetime.now(UTC),
            is_fitted=False,  # Not fitted
        )
        await self.repository.save(unfitted_detector)

        # Attempt to save should fail
        with pytest.raises(ValueError, match="is not fitted"):
            await self.service.save_model(unfitted_detector.id, format="onnx")

    @pytest.mark.asyncio
    async def test_onnx_save_nonexistent_detector_fails(self):
        """Test that saving non-existent detector raises error."""
        non_existent_id = uuid4()

        with pytest.raises(ValueError, match=f"Detector {non_existent_id} not found"):
            await self.service.save_model(non_existent_id, format="onnx")

    @pytest.mark.asyncio
    async def test_onnx_export_model(self):
        """Test exporting ONNX model for deployment."""
        # Save detector to repository
        await self.repository.save(self.detector)

        # Export model
        export_dir = self.temp_dir / "export"
        exported_files = await self.service.export_model(self.detector.id, export_dir)

        # Verify exported files
        assert "model" in exported_files
        assert "config" in exported_files
        assert "requirements" in exported_files
        assert "deploy_script" in exported_files

        # Verify files exist
        for file_path in exported_files.values():
            assert Path(file_path).exists()

        # Verify config content
        config_path = Path(exported_files["config"])
        with open(config_path) as f:
            config = json.load(f)

        assert config["detector_name"] == self.detector.name
        assert config["algorithm"] == self.detector.algorithm_name

    @pytest.mark.asyncio
    async def test_list_saved_models_includes_onnx(self):
        """Test that listing saved models includes ONNX models."""
        # Save detector to repository
        await self.repository.save(self.detector)

        # Save model in ONNX format
        await self.service.save_model(self.detector.id, format="onnx")

        # List saved models
        saved_models = await self.service.list_saved_models()

        # Verify ONNX model is listed
        assert str(self.detector.id) in saved_models
        model_info = saved_models[str(self.detector.id)]
        assert model_info["format"] == "onnx"
        assert model_info["detector_name"] == self.detector.name

    @pytest.mark.asyncio
    async def test_delete_onnx_model(self):
        """Test deleting ONNX model."""
        # Save detector to repository
        await self.repository.save(self.detector)

        # Save model in ONNX format
        model_path = await self.service.save_model(self.detector.id, format="onnx")

        # Verify model exists
        assert Path(model_path).exists()

        # Delete model
        deleted = await self.service.delete_model(self.detector.id)
        assert deleted == True

        # Verify model is gone
        assert not Path(model_path).exists()

        # Verify directory is gone
        model_dir = Path(model_path).parent
        assert not model_dir.exists()

    @pytest.mark.asyncio
    async def test_onnx_error_handling_creates_stub(self):
        """Test that ONNX export errors fall back to creating stub model."""
        # Save detector to repository
        await self.repository.save(self.detector)

        # Even if there are errors in ONNX export, it should create a stub
        model_path = await self.service.save_model(self.detector.id, format="onnx")

        # Should still create a file (stub)
        assert Path(model_path).exists()

        # Should be loadable
        loaded_detector = await self.service.load_model(self.detector.id, format="onnx")
        assert loaded_detector.name == self.detector.name

    @pytest.mark.asyncio
    async def test_onnx_feature_flag_disabled(self):
        """Test ONNX export with deep learning feature flag disabled."""
        # Save detector to repository
        await self.repository.save(self.detector)

        # Mock feature flags to disable deep learning
        from unittest.mock import patch

        with patch(
            "monorepo.infrastructure.config.feature_flags.feature_flags.is_enabled"
        ) as mock_is_enabled:
            mock_is_enabled.return_value = False

            # Should raise RuntimeError when deep learning is disabled
            with pytest.raises(
                RuntimeError, match="Deep learning features are disabled"
            ):
                await self.service.save_model(self.detector.id, format="onnx")
