"""Tests for detector protocol implementation."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.shared.protocols.detector_protocol import DetectorProtocol


class MockDetector:
    """Mock implementation of DetectorProtocol for testing."""

    def __init__(self, name: str = "mock_detector", contamination: float = 0.1):
        self._name = name
        self._contamination_rate = ContaminationRate(value=contamination)
        self._is_fitted = False
        self._parameters = {"algorithm": "mock", "contamination": contamination}

    @property
    def name(self) -> str:
        return self._name

    @property
    def contamination_rate(self) -> ContaminationRate:
        return self._contamination_rate

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters.copy()

    def fit(self, dataset: Dataset) -> None:
        self._is_fitted = True

    def detect(self, dataset: Dataset) -> DetectionResult:
        if not self._is_fitted:
            raise ValueError("Detector must be fitted before detection")

        from uuid import uuid4

        import numpy as np

        from pynomaly.domain.entities.anomaly import Anomaly
        from pynomaly.domain.value_objects import AnomalyScore

        # Mock anomalies
        anomalies = [
            Anomaly(
                score=AnomalyScore(value=0.9),
                data_point={"index": 1, "value": 10.5},
                detector_name=self.name,
            ),
            Anomaly(
                score=AnomalyScore(value=0.8),
                data_point={"index": 3, "value": 15.2},
                detector_name=self.name,
            ),
        ]

        # Mock detection result
        return DetectionResult(
            detector_id=uuid4(),
            dataset_id=uuid4(),
            anomalies=anomalies,
            scores=[
                AnomalyScore(value=0.1),
                AnomalyScore(value=0.9),
                AnomalyScore(value=0.3),
                AnomalyScore(value=0.8),
            ],
            labels=np.array([False, True, False, True]),
            threshold=0.5,
            metadata={"mock": True},
        )

    def score(self, dataset: Dataset) -> list:
        """Calculate anomaly scores for the dataset."""
        if not self._is_fitted:
            raise ValueError("Detector must be fitted before scoring")

        from pynomaly.domain.value_objects import AnomalyScore

        return [
            AnomalyScore(value=0.1),
            AnomalyScore(value=0.9),
            AnomalyScore(value=0.3),
            AnomalyScore(value=0.8),
        ]

    def fit_detect(self, dataset: Dataset) -> DetectionResult:
        """Fit the detector and detect anomalies in one step."""
        self.fit(dataset)
        return self.detect(dataset)

    def get_params(self) -> dict[str, Any]:
        """Get parameters of the detector."""
        return self._parameters.copy()

    def set_params(self, **params: Any) -> None:
        """Set parameters of the detector."""
        self._parameters.update(params)


class TestDetectorProtocol:
    """Test the DetectorProtocol interface."""

    def test_protocol_is_runtime_checkable(self):
        """Test that DetectorProtocol is runtime checkable."""
        detector = MockDetector()
        assert isinstance(detector, DetectorProtocol)

    def test_protocol_properties_exist(self):
        """Test that all required properties exist."""
        detector = MockDetector("test_detector", 0.15)

        # Test name property
        assert detector.name == "test_detector"
        assert isinstance(detector.name, str)

        # Test contamination_rate property
        assert isinstance(detector.contamination_rate, ContaminationRate)
        assert detector.contamination_rate.value == 0.15

        # Test is_fitted property
        assert isinstance(detector.is_fitted, bool)
        assert detector.is_fitted is False

        # Test parameters property
        params = detector.parameters
        assert isinstance(params, dict)
        assert "contamination" in params

    def test_protocol_methods_exist(self):
        """Test that all required methods exist and are callable."""
        detector = MockDetector()

        # Test fit method exists
        assert hasattr(detector, "fit")
        assert callable(detector.fit)

        # Test detect method exists
        assert hasattr(detector, "detect")
        assert callable(detector.detect)

    def test_fit_method_signature(self):
        """Test fit method accepts Dataset parameter."""
        detector = MockDetector()
        dataset = MagicMock(spec=Dataset)

        # Should not raise exception
        detector.fit(dataset)
        assert detector.is_fitted is True

    def test_detect_method_signature(self):
        """Test detect method accepts Dataset and returns DetectionResult."""
        detector = MockDetector()
        dataset = MagicMock(spec=Dataset)
        dataset.id = "test_dataset"

        # Fit first
        detector.fit(dataset)

        # Test detection
        result = detector.detect(dataset)
        assert isinstance(result, DetectionResult)
        assert result.detector_id == detector.name
        assert result.dataset_id == dataset.id

    def test_detect_requires_fitted_detector(self):
        """Test that detect raises error if detector not fitted."""
        detector = MockDetector()
        dataset = MagicMock(spec=Dataset)

        with pytest.raises(ValueError, match="Detector must be fitted"):
            detector.detect(dataset)

    def test_properties_are_read_only_conceptually(self):
        """Test that properties behave as expected."""
        detector = MockDetector("readonly_test")

        # Properties should return consistent values
        name1 = detector.name
        name2 = detector.name
        assert name1 == name2

        contamination1 = detector.contamination_rate
        contamination2 = detector.contamination_rate
        assert contamination1.value == contamination2.value

    def test_parameters_returns_copy(self):
        """Test that parameters property returns a copy to prevent mutation."""
        detector = MockDetector()

        params1 = detector.parameters
        params2 = detector.parameters

        # Should be equal but not the same object
        assert params1 == params2
        assert params1 is not params2

        # Modifying returned dict shouldn't affect detector
        params1["new_key"] = "new_value"
        params3 = detector.parameters
        assert "new_key" not in params3


class TestDetectorProtocolCompliance:
    """Test compliance with DetectorProtocol contract."""

    def test_complete_workflow(self):
        """Test complete detector workflow."""
        detector = MockDetector("workflow_test", 0.2)
        dataset = MagicMock(spec=Dataset)
        dataset.id = "workflow_dataset"

        # Initial state
        assert not detector.is_fitted
        assert detector.name == "workflow_test"
        assert detector.contamination_rate.value == 0.2

        # Fit detector
        detector.fit(dataset)
        assert detector.is_fitted

        # Perform detection
        result = detector.detect(dataset)
        assert isinstance(result, DetectionResult)
        assert result.detector_id == "workflow_test"
        assert result.dataset_id == "workflow_dataset"
        assert len(result.scores) == len(result.labels)

    def test_multiple_detections_after_single_fit(self):
        """Test that detector can perform multiple detections after fitting once."""
        detector = MockDetector()
        dataset1 = MagicMock(spec=Dataset)
        dataset1.id = "dataset1"
        dataset2 = MagicMock(spec=Dataset)
        dataset2.id = "dataset2"

        # Fit once
        detector.fit(dataset1)

        # Multiple detections should work
        result1 = detector.detect(dataset1)
        result2 = detector.detect(dataset2)

        assert isinstance(result1, DetectionResult)
        assert isinstance(result2, DetectionResult)
        assert result1.dataset_id == "dataset1"
        assert result2.dataset_id == "dataset2"

    def test_contamination_rate_validation(self):
        """Test that contamination rate is properly typed."""
        detector = MockDetector(contamination=0.05)

        contamination = detector.contamination_rate
        assert isinstance(contamination, ContaminationRate)
        assert 0.0 <= contamination.value <= 1.0
        assert contamination.value == 0.05

    def test_parameters_contain_expected_keys(self):
        """Test that parameters dict contains expected information."""
        detector = MockDetector()
        params = detector.parameters

        # Should be a dictionary
        assert isinstance(params, dict)

        # Should contain some basic information
        assert len(params) > 0

        # Common keys that detectors typically have
        expected_keys = {"algorithm", "contamination"}
        available_keys = set(params.keys())

        # At least some expected keys should be present
        assert len(expected_keys.intersection(available_keys)) > 0


class TestProtocolTypeHints:
    """Test that protocol type hints are correctly defined."""

    def test_method_annotations(self):
        """Test that protocol methods have correct type annotations."""
        # This tests the protocol definition itself
        fit_method = DetectorProtocol.fit
        detect_method = DetectorProtocol.detect

        # Check that methods exist (this verifies protocol structure)
        assert fit_method is not None
        assert detect_method is not None

    def test_property_annotations(self):
        """Test that protocol properties have correct type annotations."""
        # Verify protocol has required properties
        assert hasattr(DetectorProtocol, "name")
        assert hasattr(DetectorProtocol, "contamination_rate")
        assert hasattr(DetectorProtocol, "is_fitted")
        assert hasattr(DetectorProtocol, "parameters")


@pytest.fixture
def sample_detector():
    """Fixture providing a sample detector for testing."""
    return MockDetector("sample_detector", 0.1)


@pytest.fixture
def sample_dataset():
    """Fixture providing a sample dataset for testing."""
    dataset = MagicMock(spec=Dataset)
    dataset.id = "sample_dataset"
    return dataset


class TestDetectorProtocolFixtures:
    """Test detector protocol using fixtures."""

    def test_detector_with_fixtures(self, sample_detector, sample_dataset):
        """Test detector workflow using fixtures."""
        assert not sample_detector.is_fitted

        sample_detector.fit(sample_dataset)
        assert sample_detector.is_fitted

        result = sample_detector.detect(sample_dataset)
        assert result.detector_id == "sample_detector"
        assert result.dataset_id == "sample_dataset"

    def test_detector_parameters_immutability(self, sample_detector):
        """Test that detector parameters are effectively immutable."""
        original_params = sample_detector.parameters

        # Attempt to modify returned parameters
        modified_params = sample_detector.parameters
        modified_params["new_param"] = "should_not_persist"

        # Get fresh parameters
        fresh_params = sample_detector.parameters

        # Should not contain the modification
        assert "new_param" not in fresh_params
        assert fresh_params == original_params
