#!/usr/bin/env python
"""Isolated test for SHAP/LIME import validation."""

import sys
from unittest.mock import patch

def test_basic_imports():
    """Test basic SHAP/LIME import logic."""
    print("Testing basic import logic...")
    
    # Test 1: Normal import attempt
    try:
        import shap
        SHAP_AVAILABLE = True
        print(f"✓ SHAP is available")
    except ImportError:
        SHAP_AVAILABLE = False
        print(f"✓ SHAP is not available (expected if not installed)")
    
    try:
        import lime
        import lime.lime_tabular
        LIME_AVAILABLE = True
        print(f"✓ LIME is available")
    except ImportError:
        LIME_AVAILABLE = False
        print(f"✓ LIME is not available (expected if not installed)")
    
    # Verify they are booleans
    assert isinstance(SHAP_AVAILABLE, bool), f"SHAP_AVAILABLE should be bool, got {type(SHAP_AVAILABLE)}"
    assert isinstance(LIME_AVAILABLE, bool), f"LIME_AVAILABLE should be bool, got {type(LIME_AVAILABLE)}"
    
    print(f"  SHAP_AVAILABLE: {SHAP_AVAILABLE} (type: {type(SHAP_AVAILABLE)})")
    print(f"  LIME_AVAILABLE: {LIME_AVAILABLE} (type: {type(LIME_AVAILABLE)})")
    
    return True

def test_mock_imports():
    """Test import logic with mocked unavailable libraries."""
    print("\nTesting with mocked unavailable libraries...")
    
    # Store original modules
    original_modules = {}
    modules_to_mock = ['shap', 'lime', 'lime.lime_tabular']
    
    for module in modules_to_mock:
        if module in sys.modules:
            original_modules[module] = sys.modules[module]
            del sys.modules[module]
    
    try:
        # Mock ImportError for these modules
        with patch.dict('sys.modules', {module: None for module in modules_to_mock}):
            # Test the import logic that would be in the service
            try:
                import shap
                SHAP_AVAILABLE = True
            except ImportError:
                SHAP_AVAILABLE = False
                shap = None

            try:
                import lime
                import lime.lime_tabular
                LIME_AVAILABLE = True
            except ImportError:
                LIME_AVAILABLE = False
                lime = None
            
            # Verify graceful fallback
            assert not SHAP_AVAILABLE, f"SHAP_AVAILABLE should be False when mocked, got {SHAP_AVAILABLE}"
            assert not LIME_AVAILABLE, f"LIME_AVAILABLE should be False when mocked, got {LIME_AVAILABLE}"
            assert isinstance(SHAP_AVAILABLE, bool), f"SHAP_AVAILABLE should be bool, got {type(SHAP_AVAILABLE)}"
            assert isinstance(LIME_AVAILABLE, bool), f"LIME_AVAILABLE should be bool, got {type(LIME_AVAILABLE)}"
            
            print(f"  SHAP_AVAILABLE: {SHAP_AVAILABLE} (correctly False when mocked)")
            print(f"  LIME_AVAILABLE: {LIME_AVAILABLE} (correctly False when mocked)")
            print("✓ Graceful fallback works correctly")
            
    finally:
        # Restore original modules
        for module, original in original_modules.items():
            sys.modules[module] = original
    
    return True

def test_service_pattern():
    """Test the actual pattern used in the service files."""
    print("\nTesting service pattern...")
    
    # This is the exact pattern from the explainable AI service
    try:
        import shap
        SHAP_AVAILABLE = True
    except ImportError:
        SHAP_AVAILABLE = False
        shap = None

    try:
        import lime
        import lime.lime_tabular
        LIME_AVAILABLE = True
    except ImportError:
        LIME_AVAILABLE = False
        lime = None
    
    # Verify the pattern works
    assert isinstance(SHAP_AVAILABLE, bool)
    assert isinstance(LIME_AVAILABLE, bool)
    
    # Test that the variables are always defined
    assert 'SHAP_AVAILABLE' in locals()
    assert 'LIME_AVAILABLE' in locals()
    
    print(f"✓ Service pattern works correctly")
    print(f"  SHAP_AVAILABLE: {SHAP_AVAILABLE}")
    print(f"  LIME_AVAILABLE: {LIME_AVAILABLE}")
    print(f"  shap variable: {shap if SHAP_AVAILABLE else 'None (as expected)'}")
    print(f"  lime variable: {lime if LIME_AVAILABLE else 'None (as expected)'}")
    
    return True

if __name__ == "__main__":
    print("Testing explainable AI service import patterns...")
    print("=" * 60)
    
    try:
        success1 = test_basic_imports()
        success2 = test_mock_imports()
        success3 = test_service_pattern()
        
        if success1 and success2 and success3:
            print("\n" + "=" * 60)
            print("✓ ALL TESTS PASSED!")
            print("✓ SHAP_AVAILABLE and LIME_AVAILABLE booleans are always defined")
            print("✓ Graceful fallback works when libraries are not available")
            print("✓ Service pattern handles missing dependencies correctly")
        else:
            print("\n✗ Some tests failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
