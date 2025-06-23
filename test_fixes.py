#!/usr/bin/env python3
"""Test script to validate core domain fixes."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_anomaly_score_fixes():
    """Test AnomalyScore fixes."""
    print("Testing AnomalyScore...")
    
    try:
        from pynomaly.domain.value_objects import AnomalyScore, ConfidenceInterval
        from pynomaly.domain.exceptions import InvalidValueError
        
        # Test basic creation
        score = AnomalyScore(value=0.85)
        assert score.value == 0.85
        assert score.is_valid()
        assert str(score) == "0.85"
        print("✅ Basic AnomalyScore creation works")
        
        # Test with confidence interval  
        confidence = ConfidenceInterval(lower=0.8, upper=0.9, confidence_level=0.95)
        score_with_ci = AnomalyScore(value=0.85, confidence_interval=confidence)
        assert score_with_ci.confidence_interval is not None
        assert score_with_ci.confidence_lower == 0.8
        assert score_with_ci.confidence_upper == 0.9
        print("✅ AnomalyScore with confidence interval works")
        
        # Test comparison
        score1 = AnomalyScore(value=0.7)
        score2 = AnomalyScore(value=0.9)
        assert score1 < score2
        assert score2 > score1
        print("✅ AnomalyScore comparison works")
        
        # Test validation
        try:
            AnomalyScore(value=-0.1)
            assert False, "Should have raised InvalidValueError"
        except InvalidValueError:
            print("✅ AnomalyScore validation works")
            
    except Exception as e:
        print(f"❌ AnomalyScore test failed: {e}")
        return False
    
    return True

def test_confidence_interval_fixes():
    """Test ConfidenceInterval fixes."""
    print("\nTesting ConfidenceInterval...")
    
    try:
        from pynomaly.domain.value_objects import ConfidenceInterval
        from pynomaly.domain.exceptions import InvalidValueError
        
        # Test basic creation
        ci = ConfidenceInterval(lower=0.7, upper=0.9, confidence_level=0.95)
        assert ci.lower == 0.7
        assert ci.upper == 0.9
        assert ci.confidence_level == 0.95
        assert ci.width() == 0.2
        assert ci.midpoint() == 0.8
        print("✅ Basic ConfidenceInterval creation works")
        
        # Test contains
        assert ci.contains(0.8)
        assert ci.contains(0.7)
        assert ci.contains(0.9)
        assert not ci.contains(0.6)
        print("✅ ConfidenceInterval contains method works")
        
        # Test validation
        try:
            ConfidenceInterval(lower=0.9, upper=0.7)
            assert False, "Should have raised InvalidValueError"
        except InvalidValueError:
            print("✅ ConfidenceInterval validation works")
            
    except Exception as e:
        print(f"❌ ConfidenceInterval test failed: {e}")
        return False
    
    return True

def test_contamination_rate_fixes():
    """Test ContaminationRate fixes."""
    print("\nTesting ContaminationRate...")
    
    try:
        from pynomaly.domain.value_objects import ContaminationRate
        from pynomaly.domain.exceptions import InvalidValueError
        
        # Test basic creation
        rate = ContaminationRate(value=0.05)
        assert rate.value == 0.05
        assert rate.as_percentage() == 5.0
        assert str(rate) == "5.0%"
        print("✅ Basic ContaminationRate creation works")
        
        # Test edge cases
        rate1 = ContaminationRate(value=0.0)
        rate2 = ContaminationRate(value=0.5)
        assert rate1.value == 0.0
        assert rate2.value == 0.5
        print("✅ ContaminationRate edge cases work")
        
        # Test validation
        try:
            ContaminationRate(value=-0.01)
            assert False, "Should have raised InvalidValueError"
        except InvalidValueError:
            print("✅ ContaminationRate validation works")
            
    except Exception as e:
        print(f"❌ ContaminationRate test failed: {e}")
        return False
    
    return True

def test_threshold_config_fixes():
    """Test ThresholdConfig fixes."""
    print("\nTesting ThresholdConfig...")
    
    try:
        from pynomaly.domain.value_objects import ThresholdConfig
        from pynomaly.domain.exceptions import InvalidValueError
        
        # Test with parameters
        config = ThresholdConfig(
            method="percentile",
            value=95.0,
            auto_adjust=True
        )
        assert config.method == "percentile"
        assert config.value == 95.0
        assert config.auto_adjust is True
        print("✅ ThresholdConfig with parameters works")
        
        # Test defaults
        config_default = ThresholdConfig()
        assert config_default.method == "contamination"
        assert config_default.value is None
        assert config_default.auto_adjust is False
        print("✅ ThresholdConfig defaults work")
        
        # Test validation
        try:
            ThresholdConfig(method="invalid_method")
            assert False, "Should have raised InvalidValueError"
        except InvalidValueError:
            print("✅ ThresholdConfig validation works")
            
    except Exception as e:
        print(f"❌ ThresholdConfig test failed: {e}")
        return False
    
    return True

def test_anomaly_entity_fixes():
    """Test Anomaly entity fixes."""
    print("\nTesting Anomaly entity...")
    
    try:
        from pynomaly.domain.entities import Anomaly
        from pynomaly.domain.value_objects import AnomalyScore
        
        # Test basic creation
        score = AnomalyScore(value=0.95)
        anomaly = Anomaly(
            score=score,
            data_point={"feature1": 10.5, "feature2": -3.2},
            detector_name="test_detector"
        )
        
        assert anomaly.score.value == 0.95
        assert anomaly.data_point == {"feature1": 10.5, "feature2": -3.2}
        assert anomaly.detector_name == "test_detector"
        assert anomaly.severity == "critical"  # score > 0.9
        print("✅ Basic Anomaly creation works")
        
        # Test validation
        try:
            Anomaly(
                score=score,
                data_point={"feature1": 1.0},
                detector_name=""
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            print("✅ Anomaly validation works")
            
    except Exception as e:
        print(f"❌ Anomaly entity test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🔍 Testing Domain Layer Fixes...")
    
    tests = [
        test_anomaly_score_fixes,
        test_confidence_interval_fixes,
        test_contamination_rate_fixes,
        test_threshold_config_fixes,
        test_anomaly_entity_fixes,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All domain layer fixes working correctly!")
        return True
    else:
        print("❌ Some tests failed - need more fixes")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)