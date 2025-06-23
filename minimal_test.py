#!/usr/bin/env python3
"""Minimal test script to validate core domain fixes."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test basic imports."""
    print("Testing imports...")
    
    try:
        # Test value objects
        from pynomaly.domain.value_objects import AnomalyScore, ConfidenceInterval, ContaminationRate, ThresholdConfig
        print("✅ Value objects import successfully")
        
        # Test exceptions
        from pynomaly.domain.exceptions import InvalidValueError
        print("✅ Exceptions import successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_confidence_interval():
    """Test ConfidenceInterval."""
    print("\nTesting ConfidenceInterval...")
    
    try:
        from pynomaly.domain.value_objects import ConfidenceInterval
        from pynomaly.domain.exceptions import InvalidValueError
        
        # Test basic creation
        ci = ConfidenceInterval(lower=0.7, upper=0.9, confidence_level=0.95)
        assert ci.lower == 0.7
        assert ci.upper == 0.9
        assert ci.confidence_level == 0.95
        
        # Test methods
        width = ci.width()
        assert abs(width - 0.2) < 1e-10  # Account for rounding
        assert ci.midpoint() == 0.8
        print("✅ ConfidenceInterval works")
        
        return True
    except Exception as e:
        print(f"❌ ConfidenceInterval test failed: {e}")
        return False

def test_anomaly_score():
    """Test AnomalyScore."""
    print("\nTesting AnomalyScore...")
    
    try:
        from pynomaly.domain.value_objects import AnomalyScore, ConfidenceInterval
        
        # Test basic creation
        score = AnomalyScore(value=0.85)
        assert score.value == 0.85
        assert score.is_valid()
        print("✅ Basic AnomalyScore creation works")
        
        # Test with confidence interval
        ci = ConfidenceInterval(lower=0.8, upper=0.9)
        score_ci = AnomalyScore(value=0.85, confidence_interval=ci)
        assert score_ci.confidence_lower == 0.8
        assert score_ci.confidence_upper == 0.9
        print("✅ AnomalyScore with CI works")
        
        return True
    except Exception as e:
        print(f"❌ AnomalyScore test failed: {e}")
        return False

def test_contamination_rate():
    """Test ContaminationRate."""
    print("\nTesting ContaminationRate...")
    
    try:
        from pynomaly.domain.value_objects import ContaminationRate
        
        rate = ContaminationRate(value=0.05)
        assert rate.value == 0.05
        assert rate.as_percentage() == 5.0
        assert str(rate) == "5.0%"
        print("✅ ContaminationRate works")
        
        return True
    except Exception as e:
        print(f"❌ ContaminationRate test failed: {e}")
        return False

def test_threshold_config():
    """Test ThresholdConfig."""  
    print("\nTesting ThresholdConfig...")
    
    try:
        from pynomaly.domain.value_objects import ThresholdConfig
        
        # Test defaults
        config = ThresholdConfig()
        assert config.method == "contamination"
        assert config.value is None
        assert config.auto_adjust is False
        print("✅ ThresholdConfig defaults work")
        
        # Test with values
        config2 = ThresholdConfig(method="percentile", value=95.0)
        assert config2.method == "percentile"
        assert config2.value == 95.0
        print("✅ ThresholdConfig with values works")
        
        return True
    except Exception as e:
        print(f"❌ ThresholdConfig test failed: {e}")
        return False

def main():
    """Run minimal tests."""
    print("🔍 Running Minimal Domain Tests...")
    
    tests = [
        test_imports,
        test_confidence_interval,
        test_anomaly_score,
        test_contamination_rate,
        test_threshold_config,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Domain layer fixes working!")
        return True
    else:
        print("❌ Some issues remain")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)