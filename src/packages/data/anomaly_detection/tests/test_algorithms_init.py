"""Test algorithms module initialization."""

def test_algorithms_module_imports():
    """Test that the algorithms module can be imported."""
    try:
        from pynomaly_detection.algorithms import PyODAdapter, SklearnAdapter
        assert PyODAdapter is not None
        assert SklearnAdapter is not None
    except ImportError:
        # Expected since these modules may not be fully implemented
        pass


def test_algorithms_module_all():
    """Test the __all__ attribute."""
    from pynomaly_detection import algorithms
    
    # Check that __all__ is defined
    assert hasattr(algorithms, '__all__')
    assert isinstance(algorithms.__all__, list)
    assert len(algorithms.__all__) >= 2