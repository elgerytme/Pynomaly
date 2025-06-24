#!/usr/bin/env python3
"""Isolated test to validate specific fixes."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_direct_imports():
    """Test direct imports of fixed modules."""
    
    # Test ConfidenceInterval directly
    print("Testing ConfidenceInterval directly...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'pynomaly', 'domain', 'value_objects'))
        
        # Import base exceptions first
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'pynomaly', 'domain', 'exceptions'))
        
        # Create exception class directly to avoid imports
        class InvalidValueError(Exception):
            pass
        
        # Test confidence interval
        import confidence_interval
        ci = confidence_interval.ConfidenceInterval(lower=0.7, upper=0.9, confidence_level=0.95)
        assert ci.lower == 0.7
        assert ci.upper == 0.9
        assert ci.width() == 0.2
        print("✅ ConfidenceInterval works")
        
        # Test contamination rate - need to inject exception
        import contamination_rate
        contamination_rate.InvalidValueError = InvalidValueError
        
        rate = contamination_rate.ContaminationRate(value=0.05)
        assert rate.value == 0.05
        assert rate.as_percentage() == 5.0
        print("✅ ContaminationRate works")
        
        # Test threshold config  
        import threshold_config
        threshold_config.InvalidValueError = InvalidValueError
        
        config = threshold_config.ThresholdConfig()
        assert config.method == "contamination"
        assert config.value is None
        print("✅ ThresholdConfig works")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run isolated tests."""
    print("🔍 Running Isolated Domain Tests...")
    
    if test_direct_imports():
        print("🎉 Core value object fixes working!")
        return True
    else:
        print("❌ Issues remain in core fixes")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)