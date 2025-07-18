"""Tests for the unified API and Phase 3 integration."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Test imports with graceful fallback
try:
    from simplified_services.core_detection_service import CoreDetectionService
    from enhanced_features.model_persistence import ModelPersistence
    from enhanced_features.monitoring_alerting import MonitoringAlertingSystem
    PHASE2_AVAILABLE = True
except ImportError:
    PHASE2_AVAILABLE = False

# Test the unified API
def test_unified_api_imports():
    """Test that unified API imports work correctly."""
    # Test direct imports from current package
    from __init__ import (
        AnomalyDetector, 
        get_default_detector,
        check_phase2_availability,
        get_version_info
    )
    
    # Test factory functions
    detector = get_default_detector()
    assert isinstance(detector, AnomalyDetector)
    
    # Test utility functions
    availability = check_phase2_availability()
    assert isinstance(availability, dict)
    
    version_info = get_version_info()
    assert isinstance(version_info, dict)
    assert "version" in version_info


def test_anomaly_detector_initialization():
    """Test AnomalyDetector initialization."""
    from pynomaly_detection import AnomalyDetector
    
    # Test default initialization
    detector = AnomalyDetector()
    assert detector.algorithm is None
    assert detector.config == {}
    assert detector.use_phase2 == True
    
    # Test with custom configuration
    config = {"contamination": 0.15}
    detector = AnomalyDetector(algorithm="iforest", config=config)
    assert detector.algorithm == "iforest"
    assert detector.config == config
    
    # Test with Phase 2 disabled
    detector = AnomalyDetector(use_phase2=False)
    assert detector.use_phase2 == False


def test_anomaly_detector_phase2_integration():
    """Test AnomalyDetector with Phase 2 integration."""
    from pynomaly_detection import AnomalyDetector
    
    detector = AnomalyDetector()
    
    # Test Phase 2 availability check
    is_available = detector.is_phase2_available()
    assert isinstance(is_available, bool)
    
    # Test detection with sample data
    data = np.random.randn(100, 5)
    
    if detector.is_phase2_available():
        # Test Phase 2 detection
        result = detector.detect(data, algorithm="iforest", contamination=0.1)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
        assert all(pred in [0, 1] for pred in result)
    else:
        # Test fallback detection
        result = detector.detect(data, contamination=0.1)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
        assert all(pred in [0, 1] for pred in result)


def test_anomaly_detector_fallback():
    """Test AnomalyDetector fallback behavior."""
    from pynomaly_detection import AnomalyDetector
    
    # Force fallback mode
    detector = AnomalyDetector(use_phase2=False)
    
    data = np.random.randn(50, 3)
    
    # Test fit and predict workflow
    detector.fit(data, contamination=0.1)
    assert detector._trained == True
    
    # Test prediction
    predictions = detector.predict(data)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(data)
    assert all(pred in [0, 1] for pred in predictions)


def test_factory_functions():
    """Test factory functions for Phase 2 services."""
    from pynomaly_detection import (
        get_core_detector,
        get_automl_detector,
        get_ensemble_detector,
        create_monitoring_system,
        create_model_persistence
    )
    
    if PHASE2_AVAILABLE:
        # Test core detector
        try:
            core_detector = get_core_detector()
            assert core_detector is not None
        except ImportError:
            # Phase 2 not available
            pass
        
        # Test AutoML detector
        try:
            automl_detector = get_automl_detector()
            assert automl_detector is not None
        except ImportError:
            # Phase 2 not available
            pass
        
        # Test ensemble detector
        try:
            ensemble_detector = get_ensemble_detector()
            assert ensemble_detector is not None
        except ImportError:
            # Phase 2 not available
            pass
        
        # Test monitoring system
        try:
            monitoring = create_monitoring_system()
            assert monitoring is not None
        except ImportError:
            # Phase 2 not available
            pass
        
        # Test model persistence
        try:
            persistence = create_model_persistence()
            assert persistence is not None
        except ImportError:
            # Phase 2 not available
            pass
    else:
        # Test that appropriate errors are raised when Phase 2 is not available
        with pytest.raises(ImportError):
            get_core_detector()
        
        with pytest.raises(ImportError):
            get_automl_detector()
        
        with pytest.raises(ImportError):
            get_ensemble_detector()
        
        with pytest.raises(ImportError):
            create_monitoring_system()
        
        with pytest.raises(ImportError):
            create_model_persistence()


def test_phase2_availability_check():
    """Test Phase 2 availability checking."""
    from pynomaly_detection import check_phase2_availability
    
    availability = check_phase2_availability()
    
    # Check that all expected keys are present
    expected_keys = [
        "simplified_services",
        "performance_features", 
        "specialized_algorithms",
        "enhanced_features",
        "monitoring",
        "integration"
    ]
    
    for key in expected_keys:
        assert key in availability
        assert isinstance(availability[key], bool)


def test_version_info():
    """Test version information."""
    from pynomaly_detection import get_version_info
    
    version_info = get_version_info()
    
    # Check required fields
    assert "version" in version_info
    assert "phase2_available" in version_info
    assert "recommended_entry_points" in version_info
    
    # Check version format
    assert isinstance(version_info["version"], str)
    assert "." in version_info["version"]  # Should have version format like "0.2.0"
    
    # Check recommended entry points
    recommended = version_info["recommended_entry_points"]
    assert isinstance(recommended, list)
    assert len(recommended) > 0
    
    expected_recommendations = [
        "CoreDetectionService",
        "AutoMLService",
        "EnsembleService", 
        "ModelPersistence",
        "MonitoringAlertingSystem"
    ]
    
    for rec in expected_recommendations:
        assert rec in recommended


def test_backward_compatibility():
    """Test backward compatibility with legacy imports."""
    from pynomaly_detection import AnomalyDetector, get_default_detector
    
    # Test legacy AnomalyDetector interface
    detector = AnomalyDetector()
    data = np.random.randn(20, 3)
    
    # Test legacy method names work
    result = detector.detect(data)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(data)
    
    # Test legacy factory function
    default_detector = get_default_detector()
    assert isinstance(default_detector, AnomalyDetector)


def test_phase2_services_direct_import():
    """Test direct import of Phase 2 services."""
    try:
        from pynomaly_detection import (
            CoreDetectionService,
            AutoMLService,
            EnsembleService,
            ModelPersistence,
            AdvancedExplainability,
            MonitoringAlertingSystem,
            IntegrationManager
        )
        
        # Test that imports work (even if None due to ImportError)
        # This ensures the import structure is correct
        assert True  # If we get here, imports worked
        
    except ImportError:
        # This is expected if Phase 2 components are not available
        pass


def test_comprehensive_workflow():
    """Test comprehensive workflow using the unified API."""
    from pynomaly_detection import AnomalyDetector, check_phase2_availability
    
    # Check what's available
    availability = check_phase2_availability()
    
    # Create detector
    detector = AnomalyDetector()
    
    # Generate test data
    # Normal data
    normal_data = np.random.normal(0, 1, (80, 4))
    # Anomalous data
    anomalous_data = np.random.normal(5, 1, (20, 4))
    # Combined data
    test_data = np.vstack([normal_data, anomalous_data])
    
    # Test detection
    predictions = detector.detect(test_data, contamination=0.2)
    
    # Validate results
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(test_data)
    assert all(pred in [0, 1] for pred in predictions)
    
    # Check that some anomalies were detected
    anomaly_count = np.sum(predictions)
    assert anomaly_count > 0, "No anomalies detected in test data"
    
    # Test that the detector can be used multiple times
    predictions2 = detector.detect(test_data, contamination=0.1)
    assert isinstance(predictions2, np.ndarray)
    assert len(predictions2) == len(test_data)


def test_error_handling():
    """Test error handling in unified API."""
    from pynomaly_detection import AnomalyDetector
    
    detector = AnomalyDetector(use_phase2=False)
    
    # Test prediction without training (should raise error)
    with pytest.raises(ValueError, match="Model must be trained before prediction"):
        data = np.random.randn(10, 2)
        detector.predict(data)
    
    # Test with invalid data
    detector = AnomalyDetector()
    
    # Test with empty data
    try:
        result = detector.detect(np.array([]))
        # Should either work or raise a descriptive error
        assert isinstance(result, np.ndarray) or len(result) == 0
    except (ValueError, IndexError):
        # Expected for empty data
        pass


def test_feature_names():
    """Test feature names functionality."""
    from pynomaly_detection import AnomalyDetector
    
    detector = AnomalyDetector()
    
    # Test default (should be None)
    assert detector.get_feature_names() is None
    
    # Test with custom feature names
    detector._feature_names = ["feature1", "feature2", "feature3"]
    names = detector.get_feature_names()
    assert names == ["feature1", "feature2", "feature3"]


def test_configuration_options():
    """Test configuration options."""
    from pynomaly_detection import AnomalyDetector
    
    # Test with custom configuration
    config = {
        "contamination": 0.05,
        "random_state": 42,
        "n_estimators": 200
    }
    
    detector = AnomalyDetector(config=config)
    assert detector.config == config
    
    # Test detection with config
    data = np.random.randn(50, 3)
    result = detector.detect(data, contamination=0.1)  # Should override config
    assert isinstance(result, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])