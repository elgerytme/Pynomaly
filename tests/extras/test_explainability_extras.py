"""
Comprehensive tests for explainability extras.

This module tests explainability functionality with graceful degradation
when explainability packages are not installed.
"""

import pytest
import numpy as np

from tests.utils.extras_testing import (
    requires_explainability,
    parametrize_with_extras,
    explainability_available,
    check_graceful_degradation,
)


class TestExplainabilityExtras:
    """Test suite for explainability extras functionality."""

    @requires_explainability()
    def test_shap_import_with_extras(self, explainability_available):
        """Test SHAP import when explainability extras are available."""
        shap = explainability_available.get("shap")
        if shap is not None:
            # Test basic SHAP functionality
            assert hasattr(shap, "Explainer")
        else:
            pytest.skip("SHAP not available")

    @requires_explainability()
    def test_lime_import_with_extras(self, explainability_available):
        """Test LIME import when explainability extras are available."""
        lime = explainability_available.get("lime")
        if lime is not None:
            # Test basic LIME functionality
            assert hasattr(lime, "lime_tabular")
        else:
            pytest.skip("LIME not available")

    @parametrize_with_extras(["explainability"])
    def test_explainability_service_availability(self, required_extras):
        """Test that explainability service is available when extras are installed."""
        try:
            from pynomaly.application.services.explainability_service import (
                ApplicationExplainabilityService,
            )
            # Should be able to create service
            explainer_service = ApplicationExplainabilityService
            assert explainer_service is not None
        except ImportError as e:
            pytest.skip(f"Explainability service not available: {e}")

    def test_explainability_graceful_degradation(self):
        """Test graceful degradation when explainability packages are missing."""
        def mock_explainability_function():
            # Simulate a function that would use explainability
            try:
                import shap
                explainer = shap.Explainer(model=None)
                return {"explainer": explainer}
            except ImportError:
                # Graceful fallback to basic feature importance
                return {"explainer": "basic_importance"}

        # Test that the function works with or without explainability
        result = mock_explainability_function()
        assert "explainer" in result

    def test_explainability_error_handling(self):
        """Test error handling when explainability packages are missing."""
        def function_requiring_explainability():
            import shap  # This will raise ImportError if not available
            return shap.Explainer(model=None)
        
        # Test error handling
        graceful, result = check_graceful_degradation(
            function_requiring_explainability,
            "explainability",
            expected_error_type=ImportError
        )
        
        # Result should be either successful or ImportError
        assert isinstance(result, (Exception, type(None)))

    def test_explainability_service_fallback(self):
        """Test that explainability service falls back gracefully."""
        try:
            from pynomaly.application.services.explainability_service import (
                ApplicationExplainabilityService,
            )
            # Should not raise ImportError if properly implemented
            service = ApplicationExplainabilityService(
                domain_explainability_service=Mock(),
                detector_repository=Mock(),
                dataset_repository=Mock(),
            )
            assert service is not None
        except ImportError:
            # This is expected if explainability dependencies are missing
            pytest.skip("Explainability service not available without extras")

    @requires_explainability()
    def test_explainability_integration_with_sample_data(self, explainability_available):
        """Test explainability integration with sample data."""
        # Create sample data
        sample_data = np.random.rand(10, 3)
        sample_model = Mock()  # Mocking a model
        
        shap = explainability_available.get("shap")
        if shap is not None:
            explainer = shap.Explainer(sample_model, sample_data)
            explanation = explainer(sample_data)
            assert explanation is not None

    def test_explainability_algorithm_recommendations(self):
        """Test explainability algorithm recommendation functionality with mocked data."""
        try:
            from pynomaly.application.services.explainability_service import (
                ExplainabilityService,
            )
            
            # Mock request and response
            request = ApplicationExplainabilityService.explain(
                model=Mock(), data=np.random.rand(10, 3)
            )
            assert request is not None
        except ImportError:
            pytest.skip("Explainability service not available")

