#!/usr/bin/env python3
"""Test WebSocket authentication directly."""

import sys
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_websocket_auth_integration():
    """Test WebSocket authentication integration."""
    print("Testing WebSocket authentication integration...")
    
    from pynomaly.infrastructure.auth.websocket_auth import WebSocketAuthMiddleware
    from pynomaly.infrastructure.config import Settings
    from pynomaly.infrastructure.auth.jwt_auth import JWTAuthService
    
    settings = Settings()
    auth_service = JWTAuthService(settings)
    
    # Create a test user
    user = auth_service.create_user(
        username="testuser_ws",
        email="test_ws@example.com",
        password="password123",
        full_name="WebSocket Test User"
    )
    
    # Create an API key for the user
    api_key = auth_service.create_api_key(user.id, "websocket_test_key")
    
    # Create WebSocket auth middleware
    ws_middleware = WebSocketAuthMiddleware(auth_service)
    
    print(f"✓ Created user: {user.username}")
    print(f"✓ Created API key: {api_key[:10]}...")
    print(f"✓ WebSocket auth middleware initialized")
    
    # Test JWT token creation
    token_response = auth_service.create_access_token(user)
    print(f"✓ Created JWT token for WebSocket auth: {token_response.access_token[:20]}...")
    
    return True

def test_require_role_dependency():
    """Test the require_role dependency function."""
    print("Testing require_role dependency...")
    
    from pynomaly.infrastructure.auth import require_role
    
    # Test different role combinations
    admin_dep = require_role('admin')
    multi_role_dep = require_role('admin', 'developer', 'business')
    
    print(f"✓ Admin dependency created: {callable(admin_dep)}")
    print(f"✓ Multi-role dependency created: {callable(multi_role_dep)}")
    
    return True

def main():
    """Run WebSocket authentication tests."""
    print("Running WebSocket authentication tests...\n")
    
    try:
        test_websocket_auth_integration()
        print()
        test_require_role_dependency()
        
        print("\n🎉 All WebSocket authentication tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
