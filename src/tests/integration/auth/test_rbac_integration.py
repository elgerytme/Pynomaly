"""Integration tests for RBAC system with FastAPI dependencies."""

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from unittest.mock import Mock

from pynomaly.infrastructure.auth.enhanced_dependencies import (
    get_current_user,
    get_current_active_user,
    require_permissions,
    require_roles,
    require_superuser,
    require_api_key,
    require_role_or_api_key,
    require_permissions_or_api_key,
)
from pynomaly.infrastructure.auth.jwt_auth_enhanced import (
    EnhancedJWTAuthService,
    init_auth,
)
from pynomaly.infrastructure.config import Settings


class TestRBACIntegration:
    """Integration tests for RBAC system."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            secret_key="test-secret-key",
            jwt_algorithm="HS256",
            jwt_expiration=3600,
            app=Mock(environment="testing")
        )

    @pytest.fixture
    def auth_service(self, settings):
        """Create and initialize auth service."""
        return init_auth(settings)

    @pytest.fixture
    def app(self, auth_service):
        """Create test FastAPI app with auth endpoints."""
        app = FastAPI()

        @app.get("/public")
        async def public_endpoint():
            return {"message": "Public endpoint"}

        @app.get("/protected")
        async def protected_endpoint(user=Depends(get_current_user)):
            return {"message": "Protected endpoint", "user": user.username}

        @app.get("/active-only")
        async def active_only_endpoint(user=Depends(get_current_active_user)):
            return {"message": "Active user endpoint", "user": user.username}

        @app.get("/admin-only")
        async def admin_only_endpoint(user=Depends(require_roles("admin"))):
            return {"message": "Admin endpoint", "user": user.username}

        @app.get("/detectors")
        async def detectors_endpoint(user=Depends(require_permissions("detectors:read"))):
            return {"message": "Detectors endpoint", "user": user.username}

        @app.get("/manage-users")
        async def manage_users_endpoint(user=Depends(require_permissions("users:write"))):
            return {"message": "Manage users endpoint", "user": user.username}

        @app.get("/superuser-only")
        async def superuser_only_endpoint(user=Depends(require_superuser)):
            return {"message": "Superuser endpoint", "user": user.username}

        @app.get("/api-key-only")
        async def api_key_only_endpoint(user=Depends(require_api_key)):
            return {"message": "API key endpoint", "user": user.username}

        @app.get("/admin-or-api-key")
        async def admin_or_api_key_endpoint(user=Depends(require_role_or_api_key("admin"))):
            return {"message": "Admin or API key endpoint", "user": user.username}

        @app.get("/read-or-api-key")
        async def read_or_api_key_endpoint(user=Depends(require_permissions_or_api_key("detectors:read"))):
            return {"message": "Read or API key endpoint", "user": user.username}

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def regular_user(self, auth_service):
        """Create regular user."""
        return auth_service.create_user(
            username="user",
            email="user@example.com",
            password="password123",
            roles=["user"]
        )

    @pytest.fixture
    def admin_user(self, auth_service):
        """Create admin user."""
        return auth_service.create_user(
            username="admin",
            email="admin@example.com",
            password="password123",
            roles=["admin"]
        )

    @pytest.fixture
    def superuser(self, auth_service):
        """Create superuser."""
        user = auth_service.create_user(
            username="superuser",
            email="superuser@example.com",
            password="password123",
            roles=["admin"]
        )
        user.is_superuser = True
        return user

    @pytest.fixture
    def regular_token(self, auth_service, regular_user):
        """Create token for regular user."""
        return auth_service.create_access_token(regular_user).access_token

    @pytest.fixture
    def admin_token(self, auth_service, admin_user):
        """Create token for admin user."""
        return auth_service.create_access_token(admin_user).access_token

    @pytest.fixture
    def superuser_token(self, auth_service, superuser):
        """Create token for superuser."""
        return auth_service.create_access_token(superuser).access_token

    @pytest.fixture
    def api_key(self, auth_service, regular_user):
        """Create API key for regular user."""
        return auth_service.create_api_key(regular_user.id, "test-key")

    def test_public_endpoint(self, client):
        """Test public endpoint access."""
        response = client.get("/public")
        assert response.status_code == 200
        assert response.json() == {"message": "Public endpoint"}

    def test_protected_endpoint_no_auth(self, client):
        """Test protected endpoint without authentication."""
        response = client.get("/protected")
        assert response.status_code == 401

    def test_protected_endpoint_with_auth(self, client, regular_token):
        """Test protected endpoint with authentication."""
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {regular_token}"}
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Protected endpoint"
        assert response.json()["user"] == "user"

    def test_protected_endpoint_invalid_token(self, client):
        """Test protected endpoint with invalid token."""
        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401

    def test_active_only_endpoint_inactive_user(self, client, auth_service):
        """Test active-only endpoint with inactive user."""
        user = auth_service.create_user(
            username="inactive",
            email="inactive@example.com",
            password="password123"
        )
        user.is_active = False
        
        token = auth_service.create_access_token(user).access_token
        
        response = client.get(
            "/active-only",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 400

    def test_role_based_access_admin(self, client, admin_token):
        """Test role-based access for admin."""
        response = client.get(
            "/admin-only",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Admin endpoint"

    def test_role_based_access_denied(self, client, regular_token):
        """Test role-based access denied for regular user."""
        response = client.get(
            "/admin-only",
            headers={"Authorization": f"Bearer {regular_token}"}
        )
        assert response.status_code == 403

    def test_permission_based_access_allowed(self, client, regular_token):
        """Test permission-based access allowed."""
        response = client.get(
            "/detectors",
            headers={"Authorization": f"Bearer {regular_token}"}
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Detectors endpoint"

    def test_permission_based_access_denied(self, client, regular_token):
        """Test permission-based access denied."""
        response = client.get(
            "/manage-users",
            headers={"Authorization": f"Bearer {regular_token}"}
        )
        assert response.status_code == 403

    def test_superuser_access_allowed(self, client, superuser_token):
        """Test superuser access allowed."""
        response = client.get(
            "/superuser-only",
            headers={"Authorization": f"Bearer {superuser_token}"}
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Superuser endpoint"

    def test_superuser_access_denied(self, client, admin_token):
        """Test superuser access denied for admin."""
        response = client.get(
            "/superuser-only",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 403

    def test_api_key_access(self, client, api_key):
        """Test API key access."""
        response = client.get(
            "/api-key-only",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        assert response.status_code == 200
        assert response.json()["message"] == "API key endpoint"

    def test_api_key_access_invalid(self, client):
        """Test API key access with invalid key."""
        response = client.get(
            "/api-key-only",
            headers={"Authorization": "Bearer pyn_invalid_key"}
        )
        assert response.status_code == 401

    def test_role_or_api_key_with_role(self, client, admin_token):
        """Test role or API key access with role."""
        response = client.get(
            "/admin-or-api-key",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Admin or API key endpoint"

    def test_role_or_api_key_with_api_key(self, client, api_key):
        """Test role or API key access with API key."""
        response = client.get(
            "/admin-or-api-key",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Admin or API key endpoint"

    def test_role_or_api_key_denied(self, client, regular_token):
        """Test role or API key access denied."""
        response = client.get(
            "/admin-or-api-key",
            headers={"Authorization": f"Bearer {regular_token}"}
        )
        assert response.status_code == 403

    def test_permission_or_api_key_with_permission(self, client, regular_token):
        """Test permission or API key access with permission."""
        response = client.get(
            "/read-or-api-key",
            headers={"Authorization": f"Bearer {regular_token}"}
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Read or API key endpoint"

    def test_permission_or_api_key_with_api_key(self, client, api_key):
        """Test permission or API key access with API key."""
        response = client.get(
            "/read-or-api-key",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        assert response.status_code == 200
        assert response.json()["message"] == "Read or API key endpoint"

    def test_token_blacklisting(self, client, auth_service, regular_user):
        """Test token blacklisting."""
        token = auth_service.create_access_token(regular_user).access_token
        
        # Token should work initially
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        
        # Blacklist the token
        auth_service.blacklist_token(token)
        
        # Token should not work after blacklisting
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 401

    def test_key_rotation_compatibility(self, client, auth_service, regular_user):
        """Test that existing tokens work after key rotation."""
        # Create token with current key
        token = auth_service.create_access_token(regular_user).access_token
        
        # Verify token works
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        
        # Rotate keys
        auth_service.rotate_keys()
        
        # Old token should still work
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        
        # New tokens should use new key
        new_token = auth_service.create_access_token(regular_user).access_token
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {new_token}"}
        )
        assert response.status_code == 200

    def test_admin_permissions_comprehensive(self, client, admin_token):
        """Test comprehensive admin permissions."""
        endpoints = [
            "/detectors",
            "/manage-users",
            "/admin-only",
        ]
        
        for endpoint in endpoints:
            response = client.get(
                endpoint,
                headers={"Authorization": f"Bearer {admin_token}"}
            )
            assert response.status_code == 200, f"Admin access failed for {endpoint}"

    def test_regular_user_permissions_limited(self, client, regular_token):
        """Test regular user has limited permissions."""
        # Should have access to
        allowed_endpoints = ["/detectors"]
        
        for endpoint in allowed_endpoints:
            response = client.get(
                endpoint,
                headers={"Authorization": f"Bearer {regular_token}"}
            )
            assert response.status_code == 200, f"User access failed for {endpoint}"
        
        # Should not have access to
        forbidden_endpoints = ["/manage-users", "/admin-only", "/superuser-only"]
        
        for endpoint in forbidden_endpoints:
            response = client.get(
                endpoint,
                headers={"Authorization": f"Bearer {regular_token}"}
            )
            assert response.status_code == 403, f"User access allowed for {endpoint}"


class TestJWKSEndpointIntegration:
    """Integration tests for JWKS endpoint."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            secret_key="test-secret-key",
            jwt_algorithm="HS256",
            jwt_expiration=3600,
            app=Mock(environment="testing")
        )

    @pytest.fixture
    def auth_service(self, settings):
        """Create and initialize auth service."""
        return init_auth(settings)

    @pytest.fixture
    def app(self, auth_service):
        """Create test FastAPI app with JWKS endpoint."""
        from pynomaly.presentation.api.endpoints.jwks import router
        
        app = FastAPI()
        app.include_router(router)
        
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def superuser_token(self, auth_service):
        """Create superuser token."""
        user = auth_service.create_user(
            username="superuser",
            email="superuser@example.com",
            password="password123",
            roles=["admin"]
        )
        user.is_superuser = True
        return auth_service.create_access_token(user).access_token

    def test_jwks_endpoint_public(self, client):
        """Test JWKS endpoint is publicly accessible."""
        response = client.get("/auth/.well-known/jwks.json")
        assert response.status_code == 200
        
        jwks = response.json()
        assert "keys" in jwks
        assert len(jwks["keys"]) >= 1
        
        key = jwks["keys"][0]
        assert key["kty"] == "RSA"
        assert key["use"] == "sig"
        assert key["kid"] == "1"
        assert "n" in key
        assert "e" in key

    def test_jwks_endpoint_caching(self, client):
        """Test JWKS endpoint returns proper caching headers."""
        response = client.get("/auth/.well-known/jwks.json")
        assert response.status_code == 200
        assert "Cache-Control" in response.headers
        assert "max-age=3600" in response.headers["Cache-Control"]

    def test_rotate_keys_endpoint(self, client, superuser_token):
        """Test key rotation endpoint."""
        # Get initial JWKS
        response = client.get("/auth/.well-known/jwks.json")
        initial_jwks = response.json()
        initial_key_count = len(initial_jwks["keys"])
        
        # Rotate keys
        response = client.post(
            "/auth/rotate-keys",
            headers={"Authorization": f"Bearer {superuser_token}"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Keys rotated successfully"
        assert "new_key_id" in data
        
        # Check JWKS has more keys now
        response = client.get("/auth/.well-known/jwks.json")
        new_jwks = response.json()
        assert len(new_jwks["keys"]) == initial_key_count + 1

    def test_rotate_keys_unauthorized(self, client):
        """Test key rotation requires superuser."""
        response = client.post("/auth/rotate-keys")
        assert response.status_code == 401

    def test_cleanup_keys_endpoint(self, client, superuser_token):
        """Test key cleanup endpoint."""
        # Rotate keys multiple times
        for _ in range(3):
            client.post(
                "/auth/rotate-keys",
                headers={"Authorization": f"Bearer {superuser_token}"}
            )
        
        # Check we have multiple keys
        response = client.get("/auth/.well-known/jwks.json")
        jwks = response.json()
        assert len(jwks["keys"]) >= 3
        
        # Clean up keys
        response = client.post(
            "/auth/cleanup-keys?keep_count=2",
            headers={"Authorization": f"Bearer {superuser_token}"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "Old keys cleaned up" in data["message"]
        assert data["current_key_count"] == 2
        
        # Check JWKS now has fewer keys
        response = client.get("/auth/.well-known/jwks.json")
        jwks = response.json()
        assert len(jwks["keys"]) == 2

    def test_key_info_endpoint(self, client, superuser_token):
        """Test key info endpoint."""
        response = client.get(
            "/auth/keys/info",
            headers={"Authorization": f"Bearer {superuser_token}"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "current_key_id" in data
        assert "total_keys" in data
        assert "key_ids" in data
        assert "algorithm" in data
        assert data["algorithm"] == "RS256"
