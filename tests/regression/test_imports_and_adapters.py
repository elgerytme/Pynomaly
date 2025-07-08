import pytest
from pynomaly.application.use_cases import DetectAnomaliesUseCase, DetectAnomalies
from pynomaly.infrastructure.adapters import SklearnAdapter


def test_detect_anomalies_alias():
    """Test that DetectAnomalies is an alias for DetectAnomaliesUseCase."""
    assert DetectAnomalies is DetectAnomaliesUseCase


def test_sklearn_adapter_requires_algorithm_name():
    """Test SklearnAdapter instantiation requires an algorithm_name."""
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'algorithm_name'"):
        SklearnAdapter()

    adapter = SklearnAdapter('IsolationForest')
    assert adapter.algorithm_name == 'IsolationForest'


def test_optional_imports():
    """Test optional imports for shap and lime."""
    from pynomaly.application.services.explainable_ai_service import SHAP_AVAILABLE, LIME_AVAILABLE

    assert SHAP_AVAILABLE is True
    assert LIME_AVAILABLE is True

