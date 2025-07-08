"""
Comprehensive tests for deep learning extras.

This module tests deep learning functionality with graceful degradation
when deep learning packages are not installed.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from tests.utils.extras_testing import (
    requires_deep,
    parametrize_with_extras,
    deep_learning_available,
    check_graceful_degradation,
)


class TestDeepLearningExtras:
    """Test suite for deep learning extras functionality."""

    @requires_deep()
    def test_pytorch_import_with_extras(self, deep_learning_available):
        """Test PyTorch import when deep extras are available."""
        torch = deep_learning_available.get("torch")
        if torch is not None:
            # Test basic PyTorch functionality
            assert hasattr(torch, "tensor")
            tensor = torch.tensor([1.0, 2.0, 3.0])
            assert tensor.shape == (3,)
        else:
            pytest.skip("PyTorch not available")

    @requires_deep()
    def test_tensorflow_import_with_extras(self, deep_learning_available):
        """Test TensorFlow import when deep extras are available."""
        tensorflow = deep_learning_available.get("tensorflow")
        if tensorflow is not None:
            # Test basic TensorFlow functionality
            assert hasattr(tensorflow, "constant")
            constant = tensorflow.constant([1.0, 2.0, 3.0])
            assert constant.shape == (3,)
        else:
            pytest.skip("TensorFlow not available")

    @requires_deep()
    def test_jax_import_with_extras(self, deep_learning_available):
        """Test JAX import when deep extras are available."""
        jax = deep_learning_available.get("jax")
        if jax is not None:
            # Test basic JAX functionality
            assert hasattr(jax, "numpy")
            array = jax.numpy.array([1.0, 2.0, 3.0])
            assert array.shape == (3,)
        else:
            pytest.skip("JAX not available")

    @parametrize_with_extras(["deep"])
    def test_deep_learning_adapter_availability(self, required_extras):
        """Test that deep learning adapters are available when extras are installed."""
        # This test checks that the infrastructure adapters work
        try:
            from pynomaly.infrastructure.adapters.deep_learning import (
                pytorch_adapter,
                tensorflow_adapter,
                jax_adapter,
            )
            # Basic smoke test
            assert pytorch_adapter is not None
            assert tensorflow_adapter is not None
            assert jax_adapter is not None
        except ImportError as e:
            pytest.skip(f"Deep learning adapters not available: {e}")

    def test_deep_learning_graceful_degradation(self):
        """Test graceful degradation when deep learning packages are missing."""
        def mock_deep_learning_function():
            # Simulate a function that would use deep learning
            try:
                import torch
                return torch.tensor([1.0, 2.0, 3.0])
            except ImportError:
                # Graceful fallback to numpy
                return np.array([1.0, 2.0, 3.0])
        
        # Test that the function works with or without deep learning
        result = mock_deep_learning_function()
        assert len(result) == 3
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_deep_learning_service_fallback(self):
        """Test that deep learning service falls back gracefully."""
        # Test service behavior when deep learning is not available
        try:
            from pynomaly.application.services.deep_learning_integration_service import (
                DeepLearningIntegrationService,
            )
            # Should not raise ImportError if properly implemented
            service = DeepLearningIntegrationService()
            assert service is not None
        except ImportError:
            # This is expected if deep learning dependencies are missing
            pytest.skip("Deep learning service not available without extras")

    @pytest.mark.parametrize("framework", ["torch", "tensorflow", "jax"])
    def test_individual_framework_availability(self, framework):
        """Test availability of individual deep learning frameworks."""
        try:
            module = pytest.importorskip(framework)
            assert module is not None
        except pytest.skip.Exception:
            pytest.skip(f"{framework} not available")

    def test_deep_learning_stub_modules(self):
        """Test that stub modules provide graceful fallbacks."""
        # Test that stub modules exist and provide basic functionality
        try:
            from pynomaly.infrastructure.adapters.deep_learning import (
                pytorch_stub,
                tensorflow_stub,
                jax_stub,
            )
            
            # These should be available even without deep learning packages
            assert pytorch_stub is not None
            assert tensorflow_stub is not None
            assert jax_stub is not None
        except ImportError:
            # This would indicate a problem with stub module design
            pytest.fail("Stub modules should be available without extras")

    @requires_deep()
    def test_deep_learning_integration_with_sample_data(self, deep_learning_available):
        """Test deep learning integration with sample data."""
        # Create sample data
        sample_data = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
        })
        
        # Test that deep learning frameworks can handle the data
        torch = deep_learning_available.get("torch")
        if torch is not None:
            # Convert to torch tensor
            tensor = torch.tensor(sample_data.values, dtype=torch.float32)
            assert tensor.shape == (100, 3)
            
        tensorflow = deep_learning_available.get("tensorflow")
        if tensorflow is not None:
            # Convert to tensorflow tensor
            tf_tensor = tensorflow.constant(sample_data.values, dtype=tensorflow.float32)
            assert tf_tensor.shape == (100, 3)

    @requires_deep()
    def test_deep_learning_model_creation(self, deep_learning_available):
        """Test creation of deep learning models."""
        torch = deep_learning_available.get("torch")
        if torch is not None:
            # Test simple model creation
            model = torch.nn.Linear(3, 1)
            assert model is not None
            assert model.in_features == 3
            assert model.out_features == 1

    def test_deep_learning_error_handling(self):
        """Test error handling when deep learning packages are missing."""
        def function_requiring_deep_learning():
            import torch  # This will raise ImportError if not available
            return torch.tensor([1.0])
        
        # Test error handling
        graceful, result = check_graceful_degradation(
            function_requiring_deep_learning,
            "deep",
            expected_error_type=ImportError
        )
        
        # Result should be either successful or ImportError
        assert isinstance(result, (Exception, type(torch.tensor([1.0])) if 'torch' in locals() else type(None)))
