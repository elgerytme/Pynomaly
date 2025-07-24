#!/usr/bin/env python3
"""
Quick test to validate that the anomaly_detection package can be imported
and basic functionality works after fixing the import issues.
"""

def test_basic_import():
    """Test that the package can be imported successfully."""
    try:
        import anomaly_detection
        print("✅ Successfully imported anomaly_detection package")
        print(f"   Version: {anomaly_detection.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import anomaly_detection: {e}")
        return False

def test_anomaly_entity():
    """Test that the Anomaly entity works with correct interface."""
    try:
        from anomaly_detection.domain.entities.anomaly import Anomaly, AnomalyType, AnomalySeverity
        import numpy as np
        
        # Test basic creation with correct parameters
        anomaly = Anomaly(
            index=42,
            confidence_score=0.85,
            anomaly_type=AnomalyType.POINT,
            severity=AnomalySeverity.HIGH
        )
        
        assert anomaly.index == 42
        assert anomaly.confidence_score == 0.85
        assert anomaly.anomaly_type == AnomalyType.POINT
        assert anomaly.severity == AnomalySeverity.HIGH
        
        print("✅ Anomaly entity creation works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Anomaly entity test failed: {e}")
        return False

def test_domain_services():
    """Test that domain services can be imported."""
    try:
        from anomaly_detection.domain.services.detection_service import DetectionService
        from anomaly_detection.domain.services.ensemble_service import EnsembleService
        print("✅ Domain services imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import domain services: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🔍 Running import validation tests...")
    
    tests = [
        test_basic_import,
        test_anomaly_entity,
        test_domain_services
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All import validation tests passed! Package is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Package may have remaining issues.")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)