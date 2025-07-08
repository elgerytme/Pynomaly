import pytest
from pynomaly.infrastructure.config.feature_flags import FeatureNotAvailableError


def test_missing_shap_dependency():
    """Test that missing SHAP dependency raises clear error."""
    with pytest.raises(FeatureNotAvailableError) as excinfo:
        from pynomaly.utils.dependency_stubs import require_dependency
        require_dependency("shap")
    
    assert "shap" in str(excinfo.value)
    assert "pip install" in str(excinfo.value)


def test_missing_torch_dependency():
    """Test that PyTorchAdapter raises clear error when PyTorch is missing."""
    with pytest.raises(FeatureNotAvailableError) as excinfo:
        from pynomaly.infrastructure.adapters.deep_learning.pytorch_stub import PyTorchAdapter
        PyTorchAdapter()
    
    assert "PyTorchAdapter" in str(excinfo.value)
    assert "PyTorch dependency is missing" in str(excinfo.value)
    assert "pip install torch torchvision" in str(excinfo.value)


def test_tensorflow_stub_error():
    """Test that TensorFlowAdapter raises clear error when TensorFlow is missing."""
    with pytest.raises(FeatureNotAvailableError) as excinfo:
        from pynomaly.infrastructure.adapters.deep_learning.tensorflow_stub import TensorFlowAdapter
        TensorFlowAdapter()
    
    assert "TensorFlowAdapter" in str(excinfo.value)
    assert "TensorFlow dependency is missing" in str(excinfo.value)
    assert "pip install tensorflow" in str(excinfo.value)


def test_jax_stub_error():
    """Test that JAXAdapter raises clear error when JAX is missing."""
    with pytest.raises(FeatureNotAvailableError) as excinfo:
        from pynomaly.infrastructure.adapters.deep_learning.jax_stub import JAXAdapter
        JAXAdapter()
    
    assert "JAXAdapter" in str(excinfo.value)
    assert "JAX dependency is missing" in str(excinfo.value)
    assert "pip install jax jaxlib" in str(excinfo.value)

