"""Quick validation script to test core functionality works."""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_basic_imports():
    """Test that basic imports work."""
    print("🔍 Testing basic imports...")
    
    try:
        from anomaly_detection.domain.services.detection_service import DetectionService
        from anomaly_detection.domain.services.ensemble_service import EnsembleService
        from anomaly_detection.domain.services.streaming_service import StreamingService
        print("✅ Core services import successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        from anomaly_detection.infrastructure.validation.comprehensive_validators import ComprehensiveValidator
        from anomaly_detection.infrastructure.api.response_utilities import ResponseBuilder
        print("✅ Infrastructure components import successfully")
    except ImportError as e:
        print(f"❌ Infrastructure import error: {e}")
        return False
    
    return True


def test_basic_detection():
    """Test basic detection functionality."""
    print("\n🎯 Testing basic detection...")
    
    try:
        from anomaly_detection.domain.services.detection_service import DetectionService
        
        # Create simple test data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 5))
        anomaly_data = np.random.normal(3, 0.5, (10, 5))
        test_data = np.vstack([normal_data, anomaly_data])
        
        # Test detection
        service = DetectionService()
        result = service.detect_anomalies(
            data=test_data,
            algorithm="iforest",
            contamination=0.1
        )
        
        if result.success and result.total_samples == 110:
            print("✅ Basic detection works")
            print(f"   Detected {result.anomaly_count} anomalies out of {result.total_samples} samples")
            return True
        else:
            print("❌ Detection failed or returned unexpected results")
            return False
            
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False


def test_ensemble_detection():
    """Test ensemble detection functionality."""
    print("\n🔗 Testing ensemble detection...")
    
    try:
        from anomaly_detection.domain.services.ensemble_service import EnsembleService
        
        # Create test data
        np.random.seed(123)
        test_data = np.random.rand(50, 4)
        
        # Test ensemble
        service = EnsembleService()
        result = service.detect_with_ensemble(
            data=test_data,
            algorithms=["iforest", "lof"],
            combination_method="majority",
            contamination=0.1
        )
        
        if result.success and result.total_samples == 50:
            print("✅ Ensemble detection works")
            print(f"   Algorithm: {result.algorithm}")
            return True
        else:
            print("❌ Ensemble detection failed")
            return False
            
    except Exception as e:
        print(f"❌ Ensemble test failed: {e}")
        return False


def test_streaming_service():
    """Test streaming service functionality.""" 
    print("\n🌊 Testing streaming service...")
    
    try:
        from anomaly_detection.domain.services.streaming_service import StreamingService
        
        # Create streaming service
        service = StreamingService(window_size=50, update_frequency=25)
        
        # Process some samples
        np.random.seed(456)
        for i in range(10):
            sample = np.random.rand(3)
            result = service.process_sample(sample)
            
            if not isinstance(result.predictions, np.ndarray):
                print("❌ Streaming service returned invalid result")
                return False
        
        # Get stats
        stats = service.get_streaming_stats()
        
        if stats["total_samples"] == 10:
            print("✅ Streaming service works")
            print(f"   Processed {stats['total_samples']} samples")
            return True
        else:
            print("❌ Streaming service stats incorrect")
            return False
            
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        return False


def test_validation_system():
    """Test validation system."""
    print("\n✅ Testing validation system...")
    
    try:
        from anomaly_detection.infrastructure.validation.comprehensive_validators import ComprehensiveValidator
        
        validator = ComprehensiveValidator()
        
        # Test valid data
        test_data = np.random.rand(100, 5)
        result = validator.validate_detection_request(
            data=test_data,
            algorithm="iforest",
            contamination=0.1
        )
        
        if result.is_valid:
            print("✅ Validation system works")
            print(f"   Validation passed with {len(result.warnings)} warnings")
            return True
        else:
            print("❌ Validation failed for valid data")
            print(f"   Errors: {result.errors}")
            return False
            
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False


def test_response_utilities():
    """Test API response utilities."""
    print("\n📡 Testing response utilities...")
    
    try:
        from anomaly_detection.infrastructure.api.response_utilities import ResponseBuilder
        
        builder = ResponseBuilder(request_id="test-123")
        
        # Test success response
        response = builder.success(
            data={"test": "data"},
            message="Test successful"
        )
        
        if response.success and response.request_id == "test-123":
            print("✅ Response utilities work")
            return True
        else:
            print("❌ Response utilities failed")
            return False
            
    except Exception as e:
        print(f"❌ Response utilities test failed: {e}")
        return False


def test_error_handling():
    """Test error handling system."""
    print("\n🚨 Testing error handling...")
    
    try:
        from anomaly_detection.infrastructure.logging.error_handler import ErrorHandler, InputValidationError
        
        handler = ErrorHandler()
        
        # Create test error
        test_error = InputValidationError("Test validation error")
        
        # Test error response creation
        response = handler.create_error_response(test_error)
        
        if not response["success"] and "error" in response:
            print("✅ Error handling works")
            return True
        else:
            print("❌ Error handling failed")
            return False
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def main():
    """Run quick validation tests."""
    print("🚀 Running quick validation of anomaly detection system...")
    print("="*60)
    
    test_functions = [
        test_basic_imports,
        test_basic_detection,
        test_ensemble_detection,
        test_streaming_service,
        test_validation_system,
        test_response_utilities,
        test_error_handling
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_func in test_functions:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "="*60)
    print("📊 QUICK VALIDATION SUMMARY")
    print("="*60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print(f"✅ {passed}/{total} components working correctly")
        success = True
    else:
        print("⚠️  SOME TESTS FAILED")
        print(f"✅ {passed}/{total} components working correctly")
        print(f"❌ {total - passed} components need attention")
        success = False
    
    print("\n📋 System Status:")
    if success:
        print("  • Core detection algorithms: Working ✅")
        print("  • Ensemble methods: Working ✅") 
        print("  • Streaming processing: Working ✅")
        print("  • Input validation: Working ✅")
        print("  • API responses: Working ✅")
        print("  • Error handling: Working ✅")
        print("\n🎯 System is ready for comprehensive testing!")
    else:
        print("  • Some components need fixes before full testing")
        print("\n🔧 Fix failing components before running full test suite")
    
    print("="*60)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)