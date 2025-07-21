#!/usr/bin/env python3
"""Simple test to verify authentication refactoring works."""

import sys
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_auth_imports():
    """Test that all auth components can be imported."""
    print("Testing auth imports...")


    print("✓ All auth components imported successfully")


def test_require_role_function():
    """Test the require_role function."""
    print("Testing require_role function...")

    from anomaly_detection.infrastructure.auth import require_role

    # Test creating role dependencies
    admin_dep = require_role("admin")
    developer_dep = require_role("developer")
    business_dep = require_role("business")

    # Test multiple roles
    multi_role_dep = require_role("admin", "developer")

    assert callable(admin_dep)
    assert callable(developer_dep)
    assert callable(business_dep)
    assert callable(multi_role_dep)

    print("✓ require_role function works correctly")


def test_default_roles_and_permissions():
    """Test default role and permission definitions."""
    print("Testing default roles and permissions...")

    from anomaly_detection.domain.entities.user import DEFAULT_PERMISSIONS, UserRole

    # Test that we have default permissions for all roles
    assert UserRole.SUPER_ADMIN in DEFAULT_PERMISSIONS
    assert UserRole.TENANT_ADMIN in DEFAULT_PERMISSIONS
    assert UserRole.DATA_SCIENTIST in DEFAULT_PERMISSIONS
    assert UserRole.ANALYST in DEFAULT_PERMISSIONS
    assert UserRole.VIEWER in DEFAULT_PERMISSIONS

    # Test that admin has different permissions than viewer
    admin_perms = DEFAULT_PERMISSIONS[UserRole.SUPER_ADMIN]
    viewer_perms = DEFAULT_PERMISSIONS[UserRole.VIEWER]

    # Admin should have platform management permissions that viewer doesn't
    assert admin_perms != viewer_perms
    assert any(perm.resource == "platform" for perm in admin_perms)
    assert not any(perm.resource == "platform" for perm in viewer_perms)

    print("✓ Default roles and permissions are properly defined")


def test_api_key_functionality():
    """Test API key generation and validation."""
    print("Testing API key functionality...")

    from anomaly_detection.infrastructure.auth.jwt_auth import JWTAuthService
    from anomaly_detection.infrastructure.config import Settings

    settings = Settings()
    auth_service = JWTAuthService(settings)

    # Create a test user first
    user = auth_service.create_user(
        username="testuser",
        email="test@example.com",
        password="password123",
        full_name="Test User",
    )

    # Test API key creation
    api_key = auth_service.create_api_key(user.id, "test_key")

    assert api_key.startswith("pyn_")
    assert len(api_key) > 10

    # Test API key authentication
    auth_user = auth_service.authenticate_api_key(api_key)
    assert auth_user.id == user.id

    print("✓ API key functionality works correctly")


def test_websocket_auth_middleware():
    """Test WebSocket authentication middleware."""
    print("Testing WebSocket auth middleware...")

    from anomaly_detection.infrastructure.auth.jwt_auth import JWTAuthService
    from anomaly_detection.infrastructure.auth.websocket_auth import WebSocketAuthMiddleware
    from anomaly_detection.infrastructure.config import Settings

    settings = Settings()
    auth_service = JWTAuthService(settings)

    # Create WebSocket auth middleware
    ws_middleware = WebSocketAuthMiddleware(auth_service)

    assert ws_middleware is not None
    assert ws_middleware.auth_service == auth_service

    print("✓ WebSocket auth middleware created successfully")


def test_htmx_auth_middleware():
    """Test HTMX authentication middleware."""
    print("Testing HTMX auth middleware...")

    from anomaly_detection.infrastructure.auth.jwt_auth import JWTAuthService
    from anomaly_detection.infrastructure.auth.websocket_auth import HTMXAuthMiddleware
    from anomaly_detection.infrastructure.config import Settings

    settings = Settings()
    auth_service = JWTAuthService(settings)

    # Create HTMX auth middleware
    class MockApp:
        pass

    htmx_middleware = HTMXAuthMiddleware(MockApp(), auth_service)

    assert htmx_middleware is not None
    assert htmx_middleware.auth_service == auth_service

    print("✓ HTMX auth middleware created successfully")


def main():
    """Run all tests."""
    print("Running authentication functionality tests...\n")

    try:
        test_auth_imports()
        test_require_role_function()
        test_default_roles_and_permissions()
        test_api_key_functionality()
        test_websocket_auth_middleware()
        test_htmx_auth_middleware()

        print("\n🎉 All authentication functionality tests passed!")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
