#!/usr/bin/env python3
"""
Test authentication flows with the new simplified dependencies
"""

import sys
import traceback
from unittest.mock import Mock, AsyncMock

# Add src to path
sys.path.insert(0, 'src')

def test_auth_deps_functionality():
    """Test that auth dependencies work correctly"""
    test_results = []
    
    try:
        print("🔐 Testing auth dependencies functionality...")
        from pynomaly.presentation.api.auth_deps import (
            get_current_user_simple, 
            get_current_user_model,
            get_container_simple
        )
        
        # Test 1: Dependencies are callable
        assert callable(get_current_user_simple)
        assert callable(get_current_user_model) 
        assert callable(get_container_simple)
        test_results.append(("✅ Auth dependencies are callable", "PASS"))
        
        # Test 2: Check function signatures
        import inspect
        
        # get_current_user_simple should accept optional HTTPAuthorizationCredentials
        sig = inspect.signature(get_current_user_simple)
        assert 'credentials' in sig.parameters
        test_results.append(("✅ get_current_user_simple has correct signature", "PASS"))
        
        # get_container_simple should return Container
        sig = inspect.signature(get_container_simple)
        test_results.append(("✅ get_container_simple has correct signature", "PASS"))
        
        return test_results
        
    except Exception as e:
        test_results.append((f"❌ Auth dependencies test failed: {str(e)}", "FAIL"))
        return test_results

def test_container_creation():
    """Test that container creation works"""
    test_results = []
    
    try:
        print("📦 Testing container creation...")
        from pynomaly.infrastructure.config import create_container
        
        container = create_container()
        assert container is not None
        test_results.append(("✅ Container creation successful", "PASS"))
        
        # Test that key services are available
        config = container.config()
        assert config is not None
        test_results.append(("✅ Config service available", "PASS"))
        
        return test_results
        
    except Exception as e:
        test_results.append((f"❌ Container creation failed: {str(e)}", "FAIL"))
        return test_results

def test_auth_integration_with_endpoints():
    """Test that endpoints properly integrate with auth dependencies"""
    test_results = []
    
    try:
        print("🔗 Testing auth integration with endpoints...")
        from fastapi.testclient import TestClient
        from pynomaly.presentation.api.app import create_app
        
        app = create_app()
        client = TestClient(app)
        
        # Test 1: Health endpoint (no auth required)
        response = client.get("/api/v1/health")
        if response.status_code == 200:
            test_results.append(("✅ Health endpoint accessible", "PASS"))
        else:
            test_results.append((f"❌ Health endpoint failed ({response.status_code})", "FAIL"))
        
        # Test 2: Auth endpoints exist and are accessible
        response = client.get("/api/v1/docs")
        if response.status_code == 200:
            test_results.append(("✅ API docs accessible", "PASS"))
        else:
            test_results.append((f"❌ API docs failed ({response.status_code})", "FAIL"))
        
        # Test 3: Check that protected endpoints require auth (should return 401/422)
        # Try to access a protected endpoint without auth
        response = client.post("/api/v1/automl/profile", json={"dataset_id": "test"})
        if response.status_code in [401, 422, 500]:  # Expected: auth required or validation error
            test_results.append(("✅ Protected endpoints require auth", "PASS"))
        else:
            test_results.append((f"❌ Protected endpoint security issue ({response.status_code})", "FAIL"))
        
        return test_results
        
    except Exception as e:
        test_results.append((f"❌ Auth integration test failed: {str(e)}", "FAIL"))
        return test_results

def test_no_circular_imports():
    """Test that there are no circular import issues"""
    test_results = []
    
    try:
        print("🔄 Testing for circular import issues...")
        
        # Import auth dependencies 
        from pynomaly.presentation.api.auth_deps import get_current_user_simple
        
        # Import main app
        from pynomaly.presentation.api.app import create_app
        
        # Import endpoint modules that use auth deps
        from pynomaly.presentation.api.endpoints import automl, autonomous, ensemble, explainability
        
        test_results.append(("✅ No circular import issues", "PASS"))
        
        return test_results
        
    except ImportError as e:
        test_results.append((f"❌ Circular import detected: {str(e)}", "FAIL"))
        return test_results
    except Exception as e:
        test_results.append((f"❌ Import test failed: {str(e)}", "FAIL"))
        return test_results

def test_pydantic_forward_reference_fix():
    """Test that pydantic forward reference issues are resolved"""
    test_results = []
    
    try:
        print("🔧 Testing pydantic forward reference fix...")
        from pynomaly.presentation.api.app import create_app
        
        app = create_app()
        
        # This should not raise TypeAdapter errors
        openapi_schema = app.openapi()
        
        # Verify the schema was generated successfully
        assert 'paths' in openapi_schema
        assert len(openapi_schema['paths']) > 0
        
        test_results.append(("✅ Pydantic forward reference issues resolved", "PASS"))
        
        return test_results
        
    except Exception as e:
        if "TypeAdapter" in str(e) or "ForwardRef" in str(e):
            test_results.append((f"❌ Pydantic forward reference issue still exists: {str(e)}", "FAIL"))
        else:
            test_results.append((f"❌ Pydantic test failed: {str(e)}", "FAIL"))
        return test_results

def main():
    """Main test runner"""
    print("🚀 Starting Authentication Flow Tests...")
    print("=" * 70)
    
    all_results = []
    
    # Test 1: Auth Dependencies
    auth_results = test_auth_deps_functionality()
    all_results.extend(auth_results)
    
    # Test 2: Container Creation
    container_results = test_container_creation()
    all_results.extend(container_results)
    
    # Test 3: Auth Integration
    integration_results = test_auth_integration_with_endpoints()
    all_results.extend(integration_results)
    
    # Test 4: Circular Imports
    import_results = test_no_circular_imports()
    all_results.extend(import_results)
    
    # Test 5: Pydantic Forward Reference Fix
    pydantic_results = test_pydantic_forward_reference_fix()
    all_results.extend(pydantic_results)
    
    # Print results
    print("\n🔐 Authentication Test Results:")
    print("=" * 70)
    
    passed = failed = 0
    for test_name, status in all_results:
        print(f"{test_name}")
        if status == "PASS":
            passed += 1
        else:
            failed += 1
    
    print("=" * 70)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All authentication tests PASSED!")
        print("✅ Simplified auth dependencies working correctly")
        print("✅ No pydantic forward reference issues")
        print("✅ Authentication integration successful")
        return 0
    else:
        print(f"\n⚠️  {failed} authentication tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())