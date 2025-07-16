"""Test ONNX model persistence functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from monorepo.application.services.model_persistence_service import (
    ModelPersistenceService,
)
from monorepo.domain.entities import Detector
from monorepo.domain.value_objects import ContaminationRate


class TestONNXModelPersistence:
    """Test ONNX model persistence functionality."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_detector_repository(self):
        """Create mock detector repository."""
        repo = Mock()
        repo.find_by_id = Mock()
        repo.save = Mock()
        return repo

    @pytest.fixture
    def mock_detector(self):
        """Create mock detector."""
        detector = Mock(spec=Detector)
        detector.id = uuid4()
        detector.name = "Test Detector"
        detector.algorithm_name = "IsolationForest"
        detector.is_fitted = True
        detector.parameters = {"n_estimators": 100}
        detector.contamination_rate = ContaminationRate(0.1)
        detector.requires_fitting = True
        detector.supports_streaming = False
        detector._model = None  # No PyTorch model
        detector._input_shape = (1, 5)
        return detector

    @pytest.fixture
    def persistence_service(self, mock_detector_repository, temp_storage_path):
        """Create model persistence service."""
        return ModelPersistenceService(mock_detector_repository, temp_storage_path)

    @pytest.mark.asyncio
    async def test_save_onnx_model_stub(
        self, persistence_service, mock_detector_repository, mock_detector
    ):
        """Test saving ONNX model creates stub when no PyTorch model."""
        # Setup
        mock_detector_repository.find_by_id.return_value = mock_detector

        # Test
        with patch(
            "monorepo.infrastructure.config.feature_flags.feature_flags"
        ) as mock_flags:
            mock_flags.is_enabled.return_value = True

            model_path = await persistence_service.save_model(
                mock_detector.id, format="onnx"
            )

        # Verify
        assert model_path.endswith("model.onnx")
        assert Path(model_path).exists()

        # Check stub content
        with open(model_path) as f:
            stub_data = json.load(f)

        assert stub_data["model_type"] == "stub"
        assert stub_data["detector_name"] == mock_detector.name
        assert stub_data["algorithm"] == mock_detector.algorithm_name

    @pytest.mark.asyncio
    async def test_save_onnx_model_feature_disabled(
        self, persistence_service, mock_detector_repository, mock_detector
    ):
        """Test saving ONNX model fails when deep learning disabled."""
        # Setup
        mock_detector_repository.find_by_id.return_value = mock_detector

        # Test
        with patch(
            "monorepo.infrastructure.config.feature_flags.feature_flags"
        ) as mock_flags:
            mock_flags.is_enabled.return_value = False

            with pytest.raises(
                RuntimeError, match="Deep learning features are disabled"
            ):
                await persistence_service.save_model(mock_detector.id, format="onnx")

    @pytest.mark.asyncio
    async def test_load_onnx_stub_model(self, persistence_service, temp_storage_path):
        """Test loading ONNX stub model."""
        # Setup - create a stub model
        detector_id = uuid4()
        model_dir = temp_storage_path / str(detector_id)
        model_dir.mkdir(parents=True)

        stub_data = {
            "model_type": "stub",
            "detector_name": "Test Detector",
            "algorithm": "IsolationForest",
            "parameters": {"n_estimators": 100},
            "message": "This is a stub ONNX model.",
        }

        model_path = model_dir / "model.onnx"
        with open(model_path, "w") as f:
            json.dump(stub_data, f)

        # Test
        detector = await persistence_service.load_model(detector_id, format="onnx")

        # Verify
        assert detector.name == "Test Detector"
        assert detector.algorithm_name == "IsolationForest"
        assert detector.parameters == {"n_estimators": 100}

    @pytest.mark.asyncio
    async def test_onnx_model_with_pytorch_model(
        self, persistence_service, mock_detector_repository, mock_detector
    ):
        """Test saving ONNX model with PyTorch model falls back to stub."""
        # Setup
        mock_detector_repository.find_by_id.return_value = mock_detector

        # Create a mock PyTorch model
        mock_pytorch_model = Mock()
        mock_detector._model = mock_pytorch_model

        # Test - without PyTorch available, should create stub
        with patch(
            "monorepo.infrastructure.config.feature_flags.feature_flags"
        ) as mock_flags:
            mock_flags.is_enabled.return_value = True

            model_path = await persistence_service.save_model(
                mock_detector.id, format="onnx"
            )

            # Verify stub was created since PyTorch is not available
            assert model_path.endswith("model.onnx")
            assert Path(model_path).exists()

            # Check it's a stub
            with open(model_path) as f:
                stub_data = json.load(f)
            assert stub_data["model_type"] == "stub"

    @pytest.mark.asyncio
    async def test_load_onnx_model_with_runtime(
        self, persistence_service, temp_storage_path
    ):
        """Test loading ONNX model falls back to stub loading."""
        # Setup
        detector_id = uuid4()
        model_dir = temp_storage_path / str(detector_id)
        model_dir.mkdir(parents=True)

        # Create a stub ONNX model file
        stub_data = {
            "model_type": "stub",
            "detector_name": "Test Detector",
            "algorithm": "IsolationForest",
            "parameters": {"n_estimators": 100},
            "message": "This is a stub ONNX model.",
        }

        model_path = model_dir / "model.onnx"
        with open(model_path, "w") as f:
            json.dump(stub_data, f)

        # Test - without onnxruntime available, should load as stub
        detector = await persistence_service.load_model(detector_id, format="onnx")

        # Verify
        assert detector is not None
        assert detector.name == "Test Detector"
        assert detector.algorithm_name == "IsolationForest"

    @pytest.mark.asyncio
    async def test_onnx_export_without_pytorch(
        self, persistence_service, mock_detector_repository, mock_detector
    ):
        """Test ONNX export creates stub when PyTorch is not available."""
        # Setup
        mock_detector_repository.find_by_id.return_value = mock_detector

        # Test
        with patch(
            "monorepo.infrastructure.config.feature_flags.feature_flags"
        ) as mock_flags:
            mock_flags.is_enabled.return_value = True

            # Without PyTorch installed, should create a stub
            model_path = await persistence_service.save_model(
                mock_detector.id, format="onnx"
            )

            # Verify stub was created
            assert model_path.endswith("model.onnx")
            assert Path(model_path).exists()

            # Check it's a stub
            with open(model_path) as f:
                stub_data = json.load(f)
            assert stub_data["model_type"] == "stub"

    @pytest.mark.asyncio
    async def test_onnx_metadata_saved(
        self, persistence_service, mock_detector_repository, mock_detector
    ):
        """Test that ONNX model metadata is saved correctly."""
        # Setup
        mock_detector_repository.find_by_id.return_value = mock_detector

        # Test
        with patch(
            "monorepo.infrastructure.config.feature_flags.feature_flags"
        ) as mock_flags:
            mock_flags.is_enabled.return_value = True

            model_path = await persistence_service.save_model(
                mock_detector.id, format="onnx"
            )

        # Verify metadata file exists
        model_dir = Path(model_path).parent
        metadata_path = model_dir / "metadata.json"
        assert metadata_path.exists()

        # Check metadata content
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["format"] == "onnx"
        assert metadata["detector_name"] == mock_detector.name
        assert metadata["algorithm"] == mock_detector.algorithm_name

    def test_create_dummy_input(self, persistence_service, mock_detector):
        """Test creating dummy input for ONNX export returns None when torch unavailable."""
        # Test without torch available
        dummy_input = persistence_service._create_dummy_input(mock_detector)

        # Should return None when torch is not available
        assert dummy_input is None

    def test_create_dummy_input_default_shape(self, persistence_service):
        """Test creating dummy input with default shape returns None when torch unavailable."""
        # Create detector without _input_shape attribute
        detector = Mock()
        if hasattr(detector, "_input_shape"):
            delattr(detector, "_input_shape")

        # Test without torch available
        dummy_input = persistence_service._create_dummy_input(detector)

        # Should return None when torch is not available
        assert dummy_input is None

    def test_create_dummy_input_without_torch(self, persistence_service, mock_detector):
        """Test creating dummy input when torch is not available."""
        # Test without torch available
        dummy_input = persistence_service._create_dummy_input(mock_detector)

        # Should return None when torch is not available
        assert dummy_input is None
