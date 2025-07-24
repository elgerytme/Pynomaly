#!/usr/bin/env python3
"""Comprehensive validation test for the anomaly detection package after migration."""

import sys
import numpy as np
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def test_basic_imports():
    """Test that all core imports work."""
    print("🔍 Testing core imports...")
    
    try:
        from anomaly_detection.domain.services.detection_service import DetectionService
        from anomaly_detection.domain.services.ensemble_service import EnsembleService
        from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository
        from anomaly_detection.infrastructure.logging import get_logger
        print("✅ Core imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_detection_functionality():
    """Test basic detection functionality."""
    print("🔍 Testing detection functionality...")
    
    try:
        from anomaly_detection.domain.services.detection_service import DetectionService
        
        service = DetectionService()
        
        # Create test data
        np.random.seed(42)
        test_data = np.random.rand(100, 4)
        
        # Test isolation forest
        result = service.detect_anomalies(
            data=test_data,
            algorithm="iforest",
            contamination=0.1
        )
        
        assert result.success, "Detection should succeed"
        assert result.total_samples == 100, "Should process all samples"
        assert result.anomaly_count > 0, "Should detect some anomalies"
        
        print(f"✅ Detection successful - {result.anomaly_count}/{result.total_samples} anomalies")
        return True
    except Exception as e:
        print(f"❌ Detection error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble_functionality():
    """Test ensemble detection functionality."""
    print("🔍 Testing ensemble functionality...")
    
    try:
        from anomaly_detection.domain.services.ensemble_service import EnsembleService
        
        service = EnsembleService()
        
        # Create test data
        np.random.seed(42)
        test_data = np.random.rand(50, 3)
        
        # Test ensemble detection
        result = service.detect_with_ensemble(
            data=test_data,
            algorithms=["iforest", "lof"],
            combination_method="majority",
            contamination=0.1
        )
        
        assert result.success, "Ensemble detection should succeed"
        assert result.total_samples == 50, "Should process all samples"
        
        print(f"✅ Ensemble successful - {result.anomaly_count}/{result.total_samples} anomalies")
        return True
    except Exception as e:
        print(f"❌ Ensemble error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_repository():
    """Test model repository functionality."""
    print("🔍 Testing model repository...")
    
    try:
        from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository
        import tempfile
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = ModelRepository(storage_path=temp_dir)
            
            # Test basic repository operations
            models = repo.list_models()
            
            print(f"✅ Model repository working - found {len(models)} models")
            return True
    except Exception as e:
        print(f"❌ Model repository error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_monitoring_integration():
    """Test monitoring integration."""
    print("🔍 Testing monitoring integration...")
    
    try:
        from anomaly_detection.infrastructure.monitoring import (
            get_metrics_collector,
            get_health_checker, 
            get_performance_monitor
        )
        
        # Test that monitoring functions work (they may return None if not initialized)
        metrics = get_metrics_collector()
        health = get_health_checker()
        perf = get_performance_monitor()
        
        print("✅ Monitoring integration working")
        return True
    except Exception as e:
        print(f"❌ Monitoring error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_server_imports():
    """Test server component imports."""
    print("🔍 Testing server imports...")
    
    try:
        # This will test if the server module can be imported
        from anomaly_detection import server
        print("✅ Server imports working")
        return True
    except Exception as e:
        print(f"❌ Server import error: {e}")
        return False

def main():
    """Run comprehensive validation tests."""
    print("🚀 Starting comprehensive validation after domain migration...")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_detection_functionality,
        test_ensemble_functionality,
        test_model_repository,
        test_monitoring_integration,
        test_server_imports,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Domain migration validation successful!")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)