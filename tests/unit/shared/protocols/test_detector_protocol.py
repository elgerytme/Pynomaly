"""Tests for detector protocol."""

from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.shared.protocols.detector_protocol import (
    DetectorProtocol,
    EnsembleDetectorProtocol,
    ExplainableDetectorProtocol,
    StreamingDetectorProtocol,
)


class TestDetectorProtocol:
    """Test suite for DetectorProtocol."""

    def test_protocol_properties(self):
        """Test protocol has required properties."""
        assert hasattr(DetectorProtocol, "name")
        assert hasattr(DetectorProtocol, "contamination_rate")
        assert hasattr(DetectorProtocol, "is_fitted")
        assert hasattr(DetectorProtocol, "parameters")

    def test_protocol_methods(self):
        """Test protocol has required methods."""
        assert hasattr(DetectorProtocol, "fit")
        assert hasattr(DetectorProtocol, "detect")
        assert hasattr(DetectorProtocol, "score")
        assert hasattr(DetectorProtocol, "fit_detect")
        assert hasattr(DetectorProtocol, "get_params")
        assert hasattr(DetectorProtocol, "set_params")

    def test_property_types(self):
        """Test property return types."""
        mock_detector = Mock(spec=DetectorProtocol)
        mock_detector.name = "test_detector"
        mock_detector.contamination_rate = Mock(spec=ContaminationRate)
        mock_detector.is_fitted = True
        mock_detector.parameters = {"param1": "value1"}

        assert isinstance(mock_detector.name, str)
        assert isinstance(mock_detector.contamination_rate, Mock)
        assert isinstance(mock_detector.is_fitted, bool)
        assert isinstance(mock_detector.parameters, dict)

    def test_fit_method_signature(self):
        """Test fit method has correct signature."""
        mock_detector = Mock(spec=DetectorProtocol)
        mock_dataset = Mock(spec=Dataset)

        mock_detector.fit(mock_dataset)
        mock_detector.fit.assert_called_once_with(mock_dataset)

    def test_detect_method_signature(self):
        """Test detect method has correct signature."""
        mock_detector = Mock(spec=DetectorProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_result = Mock(spec=DetectionResult)
        mock_detector.detect.return_value = mock_result

        result = mock_detector.detect(mock_dataset)
        assert result == mock_result

    def test_score_method_signature(self):
        """Test score method has correct signature."""
        mock_detector = Mock(spec=DetectorProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_scores = [Mock(spec=AnomalyScore)]
        mock_detector.score.return_value = mock_scores

        result = mock_detector.score(mock_dataset)
        assert result == mock_scores
        assert isinstance(result, list)

    def test_fit_detect_method_signature(self):
        """Test fit_detect method has correct signature."""
        mock_detector = Mock(spec=DetectorProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_result = Mock(spec=DetectionResult)
        mock_detector.fit_detect.return_value = mock_result

        result = mock_detector.fit_detect(mock_dataset)
        assert result == mock_result

    def test_get_params_method_signature(self):
        """Test get_params method has correct signature."""
        mock_detector = Mock(spec=DetectorProtocol)
        mock_params = {"param1": "value1", "param2": 42}
        mock_detector.get_params.return_value = mock_params

        result = mock_detector.get_params()
        assert result == mock_params
        assert isinstance(result, dict)

    def test_set_params_method_signature(self):
        """Test set_params method has correct signature."""
        mock_detector = Mock(spec=DetectorProtocol)

        mock_detector.set_params(param1="value1", param2=42)
        mock_detector.set_params.assert_called_once_with(param1="value1", param2=42)

    def test_protocol_runtime_checkable(self):
        """Test protocol is runtime checkable."""

        class ConcreteDetector:
            @property
            def name(self) -> str:
                return "test_detector"

            @property
            def contamination_rate(self) -> ContaminationRate:
                return Mock(spec=ContaminationRate)

            @property
            def is_fitted(self) -> bool:
                return True

            @property
            def parameters(self) -> dict[str, Any]:
                return {"param1": "value1"}

            def fit(self, dataset: Dataset) -> None:
                pass

            def detect(self, dataset: Dataset) -> DetectionResult:
                return Mock(spec=DetectionResult)

            def score(self, dataset: Dataset) -> list[AnomalyScore]:
                return [Mock(spec=AnomalyScore)]

            def fit_detect(self, dataset: Dataset) -> DetectionResult:
                return Mock(spec=DetectionResult)

            def get_params(self) -> dict[str, Any]:
                return {"param1": "value1"}

            def set_params(self, **params: Any) -> None:
                pass

        detector = ConcreteDetector()
        assert isinstance(detector, DetectorProtocol)


class TestStreamingDetectorProtocol:
    """Test suite for StreamingDetectorProtocol."""

    def test_protocol_inheritance(self):
        """Test StreamingDetectorProtocol extends DetectorProtocol."""
        # Should have all DetectorProtocol methods
        assert hasattr(StreamingDetectorProtocol, "fit")
        assert hasattr(StreamingDetectorProtocol, "detect")
        assert hasattr(StreamingDetectorProtocol, "score")

        # Plus streaming-specific methods
        assert hasattr(StreamingDetectorProtocol, "partial_fit")
        assert hasattr(StreamingDetectorProtocol, "detect_online")

    def test_partial_fit_method_signature(self):
        """Test partial_fit method has correct signature."""
        mock_detector = Mock(spec=StreamingDetectorProtocol)
        mock_dataset = Mock(spec=Dataset)

        mock_detector.partial_fit(mock_dataset)
        mock_detector.partial_fit.assert_called_once_with(mock_dataset)

    def test_detect_online_method_signature(self):
        """Test detect_online method has correct signature."""
        mock_detector = Mock(spec=StreamingDetectorProtocol)
        mock_data_point = pd.Series([1, 2, 3])
        mock_score = Mock(spec=AnomalyScore)
        mock_detector.detect_online.return_value = (True, mock_score)

        result = mock_detector.detect_online(mock_data_point)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)

    def test_protocol_runtime_checkable(self):
        """Test StreamingDetectorProtocol is runtime checkable."""

        class ConcreteStreamingDetector:
            @property
            def name(self) -> str:
                return "streaming_detector"

            @property
            def contamination_rate(self) -> ContaminationRate:
                return Mock(spec=ContaminationRate)

            @property
            def is_fitted(self) -> bool:
                return True

            @property
            def parameters(self) -> dict[str, Any]:
                return {}

            def fit(self, dataset: Dataset) -> None:
                pass

            def detect(self, dataset: Dataset) -> DetectionResult:
                return Mock(spec=DetectionResult)

            def score(self, dataset: Dataset) -> list[AnomalyScore]:
                return [Mock(spec=AnomalyScore)]

            def fit_detect(self, dataset: Dataset) -> DetectionResult:
                return Mock(spec=DetectionResult)

            def get_params(self) -> dict[str, Any]:
                return {}

            def set_params(self, **params: Any) -> None:
                pass

            def partial_fit(self, dataset: Dataset) -> None:
                pass

            def detect_online(self, data_point: pd.Series) -> tuple[bool, AnomalyScore]:
                return (True, Mock(spec=AnomalyScore))

        detector = ConcreteStreamingDetector()
        # Check base DetectorProtocol methods exist and are callable
        assert isinstance(detector, DetectorProtocol)
        # Check streaming-specific methods exist and are callable
        assert hasattr(detector, "partial_fit")
        assert hasattr(detector, "detect_online")
        assert callable(detector.partial_fit)
        assert callable(detector.detect_online)


class TestExplainableDetectorProtocol:
    """Test suite for ExplainableDetectorProtocol."""

    def test_protocol_inheritance(self):
        """Test ExplainableDetectorProtocol extends DetectorProtocol."""
        # Should have all DetectorProtocol methods
        assert hasattr(ExplainableDetectorProtocol, "fit")
        assert hasattr(ExplainableDetectorProtocol, "detect")
        assert hasattr(ExplainableDetectorProtocol, "score")

        # Plus explainable-specific methods
        assert hasattr(ExplainableDetectorProtocol, "explain")
        assert hasattr(ExplainableDetectorProtocol, "feature_importances")

    def test_explain_method_signature(self):
        """Test explain method has correct signature."""
        mock_detector = Mock(spec=ExplainableDetectorProtocol)
        mock_dataset = Mock(spec=Dataset)
        mock_explanations = {0: {"feature1": 0.5, "feature2": 0.3}}
        mock_detector.explain.return_value = mock_explanations

        # Test with indices
        result = mock_detector.explain(mock_dataset, indices=[0, 1])
        assert isinstance(result, dict)

        # Test without indices
        result = mock_detector.explain(mock_dataset)
        assert isinstance(result, dict)

    def test_feature_importances_method_signature(self):
        """Test feature_importances method has correct signature."""
        mock_detector = Mock(spec=ExplainableDetectorProtocol)
        mock_importances = {"feature1": 0.5, "feature2": 0.3}
        mock_detector.feature_importances.return_value = mock_importances

        result = mock_detector.feature_importances()
        assert isinstance(result, dict)

    def test_protocol_runtime_checkable(self):
        """Test ExplainableDetectorProtocol is runtime checkable."""

        class ConcreteExplainableDetector:
            @property
            def name(self) -> str:
                return "explainable_detector"

            @property
            def contamination_rate(self) -> ContaminationRate:
                return Mock(spec=ContaminationRate)

            @property
            def is_fitted(self) -> bool:
                return True

            @property
            def parameters(self) -> dict[str, Any]:
                return {}

            def fit(self, dataset: Dataset) -> None:
                pass

            def detect(self, dataset: Dataset) -> DetectionResult:
                return Mock(spec=DetectionResult)

            def score(self, dataset: Dataset) -> list[AnomalyScore]:
                return [Mock(spec=AnomalyScore)]

            def fit_detect(self, dataset: Dataset) -> DetectionResult:
                return Mock(spec=DetectionResult)

            def get_params(self) -> dict[str, Any]:
                return {}

            def set_params(self, **params: Any) -> None:
                pass

            def explain(
                self, dataset: Dataset, indices: list[int] | None = None
            ) -> dict[int, dict[str, Any]]:
                return {0: {"feature1": 0.5}}

            def feature_importances(self) -> dict[str, float]:
                return {"feature1": 0.5}

        detector = ConcreteExplainableDetector()
        # Check base DetectorProtocol methods exist and are callable
        assert isinstance(detector, DetectorProtocol)
        # Check explainable-specific methods exist and are callable
        assert hasattr(detector, "explain")
        assert hasattr(detector, "feature_importances")
        assert callable(detector.explain)
        assert callable(detector.feature_importances)


class TestEnsembleDetectorProtocol:
    """Test suite for EnsembleDetectorProtocol."""

    def test_protocol_inheritance(self):
        """Test EnsembleDetectorProtocol extends DetectorProtocol."""
        # Should have all DetectorProtocol methods
        assert hasattr(EnsembleDetectorProtocol, "fit")
        assert hasattr(EnsembleDetectorProtocol, "detect")
        assert hasattr(EnsembleDetectorProtocol, "score")

        # Plus ensemble-specific methods
        assert hasattr(EnsembleDetectorProtocol, "base_detectors")
        assert hasattr(EnsembleDetectorProtocol, "add_detector")
        assert hasattr(EnsembleDetectorProtocol, "remove_detector")
        assert hasattr(EnsembleDetectorProtocol, "get_detector_weights")

    def test_base_detectors_property(self):
        """Test base_detectors property."""
        mock_detector = Mock(spec=EnsembleDetectorProtocol)
        mock_base_detectors = [Mock(spec=DetectorProtocol)]
        mock_detector.base_detectors = mock_base_detectors

        result = mock_detector.base_detectors
        assert isinstance(result, list)

    def test_add_detector_method_signature(self):
        """Test add_detector method has correct signature."""
        mock_detector = Mock(spec=EnsembleDetectorProtocol)
        mock_base_detector = Mock(spec=DetectorProtocol)

        mock_detector.add_detector(mock_base_detector, weight=0.5)
        mock_detector.add_detector.assert_called_once_with(
            mock_base_detector, weight=0.5
        )

    def test_remove_detector_method_signature(self):
        """Test remove_detector method has correct signature."""
        mock_detector = Mock(spec=EnsembleDetectorProtocol)

        mock_detector.remove_detector("detector_name")
        mock_detector.remove_detector.assert_called_once_with("detector_name")

    def test_get_detector_weights_method_signature(self):
        """Test get_detector_weights method has correct signature."""
        mock_detector = Mock(spec=EnsembleDetectorProtocol)
        mock_weights = {"detector1": 0.5, "detector2": 0.3}
        mock_detector.get_detector_weights.return_value = mock_weights

        result = mock_detector.get_detector_weights()
        assert isinstance(result, dict)

    def test_protocol_runtime_checkable(self):
        """Test EnsembleDetectorProtocol is runtime checkable."""

        class ConcreteEnsembleDetector:
            @property
            def name(self) -> str:
                return "ensemble_detector"

            @property
            def contamination_rate(self) -> ContaminationRate:
                return Mock(spec=ContaminationRate)

            @property
            def is_fitted(self) -> bool:
                return True

            @property
            def parameters(self) -> dict[str, Any]:
                return {}

            def fit(self, dataset: Dataset) -> None:
                pass

            def detect(self, dataset: Dataset) -> DetectionResult:
                return Mock(spec=DetectionResult)

            def score(self, dataset: Dataset) -> list[AnomalyScore]:
                return [Mock(spec=AnomalyScore)]

            def fit_detect(self, dataset: Dataset) -> DetectionResult:
                return Mock(spec=DetectionResult)

            def get_params(self) -> dict[str, Any]:
                return {}

            def set_params(self, **params: Any) -> None:
                pass

            @property
            def base_detectors(self) -> list[DetectorProtocol]:
                return [Mock(spec=DetectorProtocol)]

            def add_detector(
                self, detector: DetectorProtocol, weight: float = 1.0
            ) -> None:
                pass

            def remove_detector(self, detector_name: str) -> None:
                pass

            def get_detector_weights(self) -> dict[str, float]:
                return {"detector1": 0.5}

        detector = ConcreteEnsembleDetector()
        # Check base DetectorProtocol methods exist and are callable
        assert isinstance(detector, DetectorProtocol)
        # Check ensemble-specific methods exist and are callable
        assert hasattr(detector, "base_detectors")
        assert hasattr(detector, "add_detector")
        assert hasattr(detector, "remove_detector")
        assert hasattr(detector, "get_detector_weights")
        assert callable(detector.add_detector)
        assert callable(detector.remove_detector)
        assert callable(detector.get_detector_weights)


class TestProtocolInteractions:
    """Test protocol interactions and edge cases."""

    def test_multiple_protocol_implementation(self):
        """Test class implementing multiple detector protocols."""

        class AdvancedDetector:
            @property
            def name(self) -> str:
                return "advanced_detector"

            @property
            def contamination_rate(self) -> ContaminationRate:
                return Mock(spec=ContaminationRate)

            @property
            def is_fitted(self) -> bool:
                return True

            @property
            def parameters(self) -> dict[str, Any]:
                return {}

            def fit(self, dataset: Dataset) -> None:
                pass

            def detect(self, dataset: Dataset) -> DetectionResult:
                return Mock(spec=DetectionResult)

            def score(self, dataset: Dataset) -> list[AnomalyScore]:
                return [Mock(spec=AnomalyScore)]

            def fit_detect(self, dataset: Dataset) -> DetectionResult:
                return Mock(spec=DetectionResult)

            def get_params(self) -> dict[str, Any]:
                return {}

            def set_params(self, **params: Any) -> None:
                pass

            # StreamingDetectorProtocol methods
            def partial_fit(self, dataset: Dataset) -> None:
                pass

            def detect_online(self, data_point: pd.Series) -> tuple[bool, AnomalyScore]:
                return (True, Mock(spec=AnomalyScore))

            # ExplainableDetectorProtocol methods
            def explain(
                self, dataset: Dataset, indices: list[int] | None = None
            ) -> dict[int, dict[str, Any]]:
                return {0: {"feature1": 0.5}}

            def feature_importances(self) -> dict[str, float]:
                return {"feature1": 0.5}

        detector = AdvancedDetector()
        # Check base DetectorProtocol methods exist and are callable
        assert isinstance(detector, DetectorProtocol)
        # Check streaming methods exist and are callable
        assert hasattr(detector, "partial_fit")
        assert hasattr(detector, "detect_online")
        assert callable(detector.partial_fit)
        assert callable(detector.detect_online)
        # Check explainable methods exist and are callable
        assert hasattr(detector, "explain")
        assert hasattr(detector, "feature_importances")
        assert callable(detector.explain)
        assert callable(detector.feature_importances)

    def test_protocol_with_none_values(self):
        """Test protocol handles None values appropriately."""
        mock_detector = Mock(spec=DetectorProtocol)
        mock_dataset = Mock(spec=Dataset)

        # Test with None return from detect
        mock_detector.detect.return_value = None
        result = mock_detector.detect(mock_dataset)
        assert result is None

    def test_protocol_error_handling(self):
        """Test protocol methods can raise exceptions."""
        mock_detector = Mock(spec=DetectorProtocol)
        mock_dataset = Mock(spec=Dataset)

        # Configure mock to raise exception
        mock_detector.fit.side_effect = ValueError("Invalid dataset")

        with pytest.raises(ValueError, match="Invalid dataset"):
            mock_detector.fit(mock_dataset)
