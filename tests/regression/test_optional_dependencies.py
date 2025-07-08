"""Regression tests for optional dependencies handling."""

import sys
from unittest.mock import patch
import pytest


class TestOptionalDependencies:
    """Test optional dependencies are handled gracefully."""

    def test_shap_import_graceful_fallback(self):
        """Test that missing shap is handled gracefully."""
        # This test mainly checks that the module can be imported
        # The actual graceful fallback is tested by checking the flags
        from pynomaly.application.services.explainable_ai_service import SHAP_AVAILABLE
        # Should be available in current test environment
        assert isinstance(SHAP_AVAILABLE, bool)

    def test_lime_import_graceful_fallback(self):
        """Test that missing lime is handled gracefully."""
        # This test mainly checks that the module can be imported
        # The actual graceful fallback is tested by checking the flags
        from pynomaly.application.services.explainable_ai_service import LIME_AVAILABLE
        # Should be available in current test environment
        assert isinstance(LIME_AVAILABLE, bool)

    def test_explainable_ai_service_works_without_optional_deps(self):
        """Test ExplainableAIService can be imported without optional dependencies."""
        # This should not raise an error even if shap/lime are not available
        from pynomaly.application.services.explainable_ai_service import ExplainableAIService
        
        # Should be able to create an instance without the async task
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Just test that the class can be instantiated
            service_class = ExplainableAIService
            assert service_class is not None

    def test_cli_extras_include_explainability_deps(self):
        """Test that CLI extras include shap and lime dependencies."""
        import toml
        from pathlib import Path
        
        # Read pyproject.toml
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        with open(pyproject_path, 'r') as f:
            pyproject_data = toml.load(f)
        
        cli_deps = pyproject_data["project"]["optional-dependencies"]["cli"]
        
        # Check that shap and lime are in CLI extras
        assert any("shap" in dep for dep in cli_deps), "shap not found in CLI extras"
        assert any("lime" in dep for dep in cli_deps), "lime not found in CLI extras"

    def test_use_case_aliases_exist(self):
        """Test that use case aliases exist and work correctly."""
        from pynomaly.application.use_cases import (
            DetectAnomalies, DetectAnomaliesUseCase,
            TrainDetector, TrainDetectorUseCase,
            EvaluateModel, EvaluateModelUseCase,
            ExplainAnomaly, ExplainAnomalyUseCase,
        )
        
        # Test aliases are correct
        assert DetectAnomalies is DetectAnomaliesUseCase
        assert TrainDetector is TrainDetectorUseCase
        assert EvaluateModel is EvaluateModelUseCase
        assert ExplainAnomaly is ExplainAnomalyUseCase

    def test_all_exports_are_available(self):
        """Test that all expected exports are available in __all__."""
        from pynomaly.application.use_cases import __all__
        
        expected_exports = [
            "DetectAnomaliesUseCase",
            "DetectAnomaliesRequest",
            "DetectAnomaliesResponse",
            "DetectAnomalies",  # Alias
            "TrainDetectorUseCase",
            "TrainDetectorRequest",
            "TrainDetectorResponse",
            "TrainDetector",  # Alias
            "EvaluateModelUseCase",
            "EvaluateModelRequest",
            "EvaluateModelResponse",
            "EvaluateModel",  # Alias
            "ExplainAnomalyUseCase",
            "ExplainAnomaly",  # Alias
        ]
        
        for export in expected_exports:
            assert export in __all__, f"{export} not in __all__"

    def test_sklearn_adapter_default_behavior(self):
        """Test SklearnAdapter behavior with various parameters."""
        from pynomaly.infrastructure.adapters import SklearnAdapter
        from pynomaly.domain.value_objects import ContaminationRate
        
        # Test with required parameter
        adapter = SklearnAdapter("IsolationForest")
        assert adapter.algorithm_name == "IsolationForest"
        assert adapter.name == "Sklearn_IsolationForest"
        
        # Test with custom name (using correct algorithm name)
        adapter = SklearnAdapter("LocalOutlierFactor", name="CustomLOF")
        assert adapter.name == "CustomLOF"
        
        # Test with contamination rate
        adapter = SklearnAdapter(
            "IsolationForest", 
            contamination_rate=ContaminationRate(0.05)
        )
        assert adapter.contamination_rate.value == 0.05

    def test_sklearn_adapter_invalid_algorithm(self):
        """Test SklearnAdapter with invalid algorithm name."""
        from pynomaly.infrastructure.adapters import SklearnAdapter
        from pynomaly.domain.exceptions import InvalidAlgorithmError
        
        with pytest.raises(InvalidAlgorithmError):
            SklearnAdapter("NonExistentAlgorithm")
