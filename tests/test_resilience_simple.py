#!/usr/bin/env python3
"""Simple test for resilience patterns."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test basic tenacity functionality
try:
    from tenacity import retry, stop_after_attempt, wait_exponential
    print("✓ Tenacity import successful")
except ImportError as e:
    print(f"✗ Tenacity import failed: {e}")
    sys.exit(1)

# Test our resilience configuration
try:
    from pynomaly.infrastructure.config.settings import Settings
    settings = Settings()
    print(f"✓ Settings loaded - resilience enabled: {settings.resilience_enabled}")
except Exception as e:
    print(f"✗ Settings loading failed: {e}")

# Test basic retry functionality
def test_basic_retry():
    """Test basic retry functionality."""
    attempt_count = 0
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.1, max=1))
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError("Simulated failure")
        return "Success after retry"
    
    try:
        result = flaky_function()
        print(f"✓ Retry test passed: {result} (attempts: {attempt_count})")
        return True
    except Exception as e:
        print(f"✗ Retry test failed: {e}")
        return False

# Test circuit breaker functionality
def test_circuit_breaker():
    """Test circuit breaker functionality."""
    try:
        from pynomaly.infrastructure.resilience.circuit_breaker import CircuitBreaker

        # Create circuit breaker
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        def failing_function():
            raise ConnectionError("Always fails")
        
        # Test that it fails normally first
        try:
            cb.call(failing_function)
        except ConnectionError:
            pass  # Expected
        
        # Second failure should open circuit
        try:
            cb.call(failing_function)
        except ConnectionError:
            pass  # Expected
        
        # Third call should be blocked
        try:
            cb.call(failing_function)
            print("✗ Circuit breaker should have blocked the call")
            return False
        except Exception as e:
            if "Circuit breaker" in str(e):
                print("✓ Circuit breaker test passed")
                return True
            else:
                print(f"✗ Unexpected error: {e}")
                return False
    except Exception as e:
        print(f"✗ Circuit breaker test failed: {e}")
        return False

# Test enhanced resilience wrapper
def test_enhanced_wrapper():
    """Test enhanced resilience wrapper."""
    try:
        from pynomaly.infrastructure.resilience.enhanced_wrapper import (
            EnhancedResilienceWrapper,
            ResilienceConfig,
        )
        
        config = ResilienceConfig(
            operation_type="test",
            max_attempts=3,
            base_delay=0.1,
            enable_circuit_breaker=False,  # Disable for simple test
        )
        wrapper = EnhancedResilienceWrapper(config)
        
        attempt_count = 0
        
        @wrapper
        def test_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary failure")
            return "Success with enhanced wrapper"
        
        result = test_function()
        print(f"✓ Enhanced wrapper test passed: {result} (attempts: {attempt_count})")
        return True
    except Exception as e:
        print(f"✗ Enhanced wrapper test failed: {e}")
        return False

# Test resilient decorator
def test_resilient_decorator():
    """Test resilient decorator."""
    try:
        from pynomaly.infrastructure.resilience.enhanced_wrapper import resilient
        
        attempt_count = 0
        
        @resilient(
            operation_type="test",
            operation_name="test_operation",
            max_attempts=2,
            base_delay=0.1,
            enable_circuit_breaker=False,
        )
        def decorated_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ConnectionError("Temporary failure")
            return "Success with decorator"
        
        result = decorated_function()
        print(f"✓ Resilient decorator test passed: {result} (attempts: {attempt_count})")
        return True
    except Exception as e:
        print(f"✗ Resilient decorator test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Resilience Patterns Test Suite ===")
    
    tests = [
        test_basic_retry,
        test_circuit_breaker,
        test_enhanced_wrapper,
        test_resilient_decorator,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print(f"\n=== Test Results: {passed}/{total} passed ===")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
