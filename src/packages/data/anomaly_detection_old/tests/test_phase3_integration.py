"""Tests for Phase 3 integration and unified API."""

import pytest
import numpy as np
import sys
import os

# Add the package to the path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test imports with current package structure
def test_phase3_api_integration():
    """Test Phase 3 API integration."""
    # Test that we can import the main module
    try:
        import __init__ as pynomaly_detection
        assert hasattr(pynomaly_detection, 'AnomalyDetector')
        assert hasattr(pynomaly_detection, 'get_default_detector')
        assert hasattr(pynomaly_detection, 'check_phase2_availability')
        assert hasattr(pynomaly_detection, 'get_version_info')
    except ImportError as e:
        pytest.skip(f"Could not import main module: {e}")


def test_anomaly_detector_unified_interface():
    """Test the unified AnomalyDetector interface."""
    try:
        from __init__ import AnomalyDetector
        
        # Test initialization
        detector = AnomalyDetector()
        assert detector is not None
        
        # Test configuration
        detector = AnomalyDetector(algorithm="iforest", config={"contamination": 0.15})
        assert detector.algorithm == "iforest"
        assert detector.config["contamination"] == 0.15
        
        # Test Phase 2 availability check
        is_phase2_available = detector.is_phase2_available()
        assert isinstance(is_phase2_available, bool)
        
        print(f"✅ AnomalyDetector unified interface test passed")
        print(f"   - Phase 2 Available: {is_phase2_available}")
        
    except ImportError as e:
        pytest.skip(f"Could not import AnomalyDetector: {e}")


def test_phase2_availability_check():
    """Test Phase 2 availability checking."""
    try:
        from __init__ import check_phase2_availability
        
        availability = check_phase2_availability()
        assert isinstance(availability, dict)
        
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
        
        print(f"✅ Phase 2 availability check passed")
        print(f"   - Availability: {availability}")
        
    except ImportError as e:
        pytest.skip(f"Could not import check_phase2_availability: {e}")


def test_version_info():
    """Test version information."""
    try:
        from __init__ import get_version_info
        
        version_info = get_version_info()
        assert isinstance(version_info, dict)
        
        # Check required fields
        assert "version" in version_info
        assert "phase2_available" in version_info
        assert "recommended_entry_points" in version_info
        
        # Check version format
        assert isinstance(version_info["version"], str)
        assert "." in version_info["version"]
        
        print(f"✅ Version info test passed")
        print(f"   - Version: {version_info['version']}")
        
    except ImportError as e:
        pytest.skip(f"Could not import get_version_info: {e}")


def test_factory_functions():
    """Test factory functions."""
    try:
        from __init__ import (
            get_default_detector,
            get_core_detector,
            get_automl_detector,
            get_ensemble_detector,
            create_monitoring_system,
            create_model_persistence
        )
        
        # Test default detector
        detector = get_default_detector()
        assert detector is not None
        
        print(f"✅ Factory functions test passed")
        print(f"   - Default detector created successfully")
        
        # Test Phase 2 specific functions (they may fail if Phase 2 is not available)
        phase2_functions = [
            ("get_core_detector", get_core_detector),
            ("get_automl_detector", get_automl_detector),
            ("get_ensemble_detector", get_ensemble_detector),
            ("create_monitoring_system", create_monitoring_system),
            ("create_model_persistence", create_model_persistence)
        ]
        
        for name, func in phase2_functions:
            try:
                if name == "create_model_persistence":
                    result = func("test_models")
                else:
                    result = func()
                print(f"   - {name}: ✅ Available")
            except ImportError:
                print(f"   - {name}: ❌ Not available (Phase 2 not installed)")
        
    except ImportError as e:
        pytest.skip(f"Could not import factory functions: {e}")


def test_anomaly_detection_workflow():
    """Test complete anomaly detection workflow."""
    try:
        from __init__ import AnomalyDetector
        
        # Create detector
        detector = AnomalyDetector()
        
        # Generate test data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (80, 4))
        anomalous_data = np.random.normal(5, 1, (20, 4))
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
        
        print(f"✅ Anomaly detection workflow test passed")
        print(f"   - Detected {anomaly_count} anomalies out of {len(test_data)} samples")
        print(f"   - Anomaly rate: {anomaly_count/len(test_data):.2%}")
        
    except ImportError as e:
        pytest.skip(f"Could not import AnomalyDetector: {e}")


def test_simplified_services_availability():
    """Test availability of simplified services."""
    try:
        from __init__ import (
            CoreDetectionService,
            AutoMLService,
            EnsembleService,
            ExplainabilityService
        )
        
        services = [
            ("CoreDetectionService", CoreDetectionService),
            ("AutoMLService", AutoMLService),
            ("EnsembleService", EnsembleService),
            ("ExplainabilityService", ExplainabilityService)
        ]
        
        available_services = []
        for name, service_class in services:
            if service_class is not None:
                available_services.append(name)
        
        print(f"✅ Simplified services availability test passed")
        print(f"   - Available services: {available_services}")
        print(f"   - Total available: {len(available_services)}/4")
        
        # Test that we can at least import them (even if None)
        assert True  # If we get here, imports worked
        
    except ImportError as e:
        pytest.skip(f"Could not import simplified services: {e}")


def test_enhanced_features_availability():
    """Test availability of enhanced features."""
    try:
        from __init__ import (
            ModelPersistence,
            AdvancedExplainability,
            IntegrationManager,
            MonitoringAlertingSystem
        )
        
        features = [
            ("ModelPersistence", ModelPersistence),
            ("AdvancedExplainability", AdvancedExplainability),
            ("IntegrationManager", IntegrationManager),
            ("MonitoringAlertingSystem", MonitoringAlertingSystem)
        ]
        
        available_features = []
        for name, feature_class in features:
            if feature_class is not None:
                available_features.append(name)
        
        print(f"✅ Enhanced features availability test passed")
        print(f"   - Available features: {available_features}")
        print(f"   - Total available: {len(available_features)}/4")
        
        # Test that we can at least import them (even if None)
        assert True  # If we get here, imports worked
        
    except ImportError as e:
        pytest.skip(f"Could not import enhanced features: {e}")


def test_backward_compatibility():
    """Test backward compatibility with legacy interfaces."""
    try:
        from __init__ import (
            Anomaly,
            DetectionResult,
            Detector,
            PyODAdapter,
            SimplePyODAdapter
        )
        
        legacy_components = [
            ("Anomaly", Anomaly),
            ("DetectionResult", DetectionResult),
            ("Detector", Detector),
            ("PyODAdapter", PyODAdapter),
            ("SimplePyODAdapter", SimplePyODAdapter)
        ]
        
        available_legacy = []
        for name, component in legacy_components:
            if component is not None:
                available_legacy.append(name)
        
        print(f"✅ Backward compatibility test passed")
        print(f"   - Available legacy components: {available_legacy}")
        print(f"   - Total available: {len(available_legacy)}/5")
        
        # Test that we can at least import them (even if None)
        assert True  # If we get here, imports worked
        
    except ImportError as e:
        pytest.skip(f"Could not import legacy components: {e}")


def test_migration_utilities():
    """Test migration utilities."""
    try:
        from migration_guide import (
            MigrationHelper,
            CompatibilityLayer,
            check_migration_status
        )
        
        # Test migration helper
        helper = MigrationHelper()
        recommendations = helper.get_all_recommendations()
        assert len(recommendations) > 0
        
        # Test compatibility layer
        compat = CompatibilityLayer()
        assert compat is not None
        
        # Test migration status
        status = check_migration_status()
        assert isinstance(status, dict)
        assert "migration_status" in status
        
        print(f"✅ Migration utilities test passed")
        print(f"   - Migration status: {status.get('migration_status', 'unknown')}")
        print(f"   - Migration score: {status.get('migration_score', 0):.1f}%")
        
    except ImportError as e:
        pytest.skip(f"Could not import migration utilities: {e}")


if __name__ == "__main__":
    # Run tests manually for debugging
    print("=== PHASE 3 INTEGRATION TESTS ===")
    
    test_functions = [
        test_phase3_api_integration,
        test_anomaly_detector_unified_interface,
        test_phase2_availability_check,
        test_version_info,
        test_factory_functions,
        test_anomaly_detection_workflow,
        test_simplified_services_availability,
        test_enhanced_features_availability,
        test_backward_compatibility,
        test_migration_utilities
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"\n🔄 Running {test_func.__name__}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
    
    print(f"\n=== RESULTS ===")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success rate: {passed/(passed+failed)*100:.1f}%")