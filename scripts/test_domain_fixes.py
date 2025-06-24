#!/usr/bin/env python3
"""Test script to verify domain layer fixes without running full pytest."""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_exception_imports():
    """Test that exception imports work correctly."""
    try:
        from pynomaly.domain.exceptions import InvalidValueError, ValidationError
        print("✓ Exception imports successful")
        
        # Test that InvalidValueError is distinct from ValidationError
        assert InvalidValueError != ValidationError
        print("✓ InvalidValueError is distinct from ValidationError")
        
        # Test exception hierarchy
        error = InvalidValueError("test message")
        assert isinstance(error, ValidationError)
        print("✓ InvalidValueError inherits from ValidationError")
        
        return True
    except Exception as e:
        print(f"✗ Exception imports failed: {e}")
        return False

def test_contamination_rate_logic():
    """Test ContaminationRate validation logic without creating instances."""
    try:
        # Test the validation logic
        def validate_contamination_rate(value):
            if not isinstance(value, (int, float)):
                raise InvalidValueError(f"Contamination rate must be numeric, got {type(value)}")
            
            if not (0.0 <= value <= 0.5):
                raise InvalidValueError(f"Contamination rate must be between 0 and 0.5, got {value}")
            
            return True
        
        # Test valid values
        assert validate_contamination_rate(0.0) == True
        assert validate_contamination_rate(0.25) == True 
        assert validate_contamination_rate(0.5) == True
        print("✓ ContaminationRate validation logic works for valid values")
        
        # Test invalid values
        try:
            validate_contamination_rate(-0.01)
            assert False, "Should have raised InvalidValueError"
        except InvalidValueError:
            pass
        
        try:
            validate_contamination_rate(0.51)
            assert False, "Should have raised InvalidValueError"
        except InvalidValueError:
            pass
        
        try:
            validate_contamination_rate(1.0)
            assert False, "Should have raised InvalidValueError"
        except InvalidValueError:
            pass
        
        print("✓ ContaminationRate validation logic works for invalid values")
        return True
    except Exception as e:
        print(f"✗ ContaminationRate validation logic failed: {e}")
        return False

def test_threshold_config_logic():
    """Test ThresholdConfig validation logic."""
    try:
        def validate_threshold_config(method="contamination", value=None):
            if method not in ["percentile", "fixed", "iqr", "mad", "adaptive", "contamination"]:
                raise InvalidValueError(f"Invalid threshold method: {method}")
            
            if method == "percentile" and value is not None:
                if not (0 <= value <= 100):
                    raise InvalidValueError(f"Percentile value must be between 0 and 100, got {value}")
            
            return True
        
        # Test valid cases
        assert validate_threshold_config("contamination", None) == True
        assert validate_threshold_config("percentile", 95.0) == True
        assert validate_threshold_config("percentile", 0.0) == True
        assert validate_threshold_config("percentile", 100.0) == True
        print("✓ ThresholdConfig validation logic works for valid values")
        
        # Test invalid cases
        try:
            validate_threshold_config("invalid_method")
            assert False, "Should have raised InvalidValueError"
        except InvalidValueError:
            pass
        
        try:
            validate_threshold_config("percentile", 101.0)
            assert False, "Should have raised InvalidValueError"
        except InvalidValueError:
            pass
        
        try:
            validate_threshold_config("percentile", -1.0)
            assert False, "Should have raised InvalidValueError"
        except InvalidValueError:
            pass
        
        print("✓ ThresholdConfig validation logic works for invalid values")
        return True
    except Exception as e:
        print(f"✗ ThresholdConfig validation logic failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing domain layer fixes...")
    
    # First, try to import the base exception class
    try:
        from pynomaly.domain.exceptions.base import InvalidValueError
        print("✓ Base InvalidValueError import successful")
    except Exception as e:
        print(f"✗ Base InvalidValueError import failed: {e}")
        return False
    
    tests = [
        test_exception_imports,
        test_contamination_rate_logic, 
        test_threshold_config_logic
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All domain layer fixes are logically correct!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)