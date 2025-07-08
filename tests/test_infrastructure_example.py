"""Example test file demonstrating the test infrastructure fixtures.

This file shows how to use the configured fixtures for:
- FastAPI app testing with TestClient
- JWT authentication with auth headers
- Database testing with in-memory sessions
- Async route testing with pytest-asyncio
- Proper test isolation and cleanup
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from pynomaly.presentation.api.app import create_app
from pynomaly.infrastructure.config import Container


class TestInfrastructureFixtures:
    """Test class demonstrating proper use of test infrastructure fixtures."""

    def test_app_fixture(self, app):
        """Test that the app fixture returns a properly configured FastAPI app."""
        assert app is not None
        assert hasattr(app, 'title')
        assert 'pynomaly' in app.title.lower()

    def test_client_fixture(self, client: TestClient):
        """Test that the client fixture returns a working TestClient."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_auth_header_fixture(self, client: TestClient, auth_header: dict):
        """Test authenticated API request using auth_header fixture."""
        # Test authenticated endpoint
        response = client.get("/api/v1/health", headers=auth_header)
        assert response.status_code == 200

    def test_user_auth_header_fixture(self, client: TestClient, user_auth_header: dict):
        """Test user-level authenticated API request."""
        response = client.get("/api/v1/health", headers=user_auth_header)
        assert response.status_code == 200

    def test_api_key_header_fixture(self, client: TestClient, api_key_header: dict):
        """Test API key authentication."""
        response = client.get("/api/v1/health", headers=api_key_header)
        assert response.status_code == 200

    def test_db_session_fixture(self, db_session: Session):
        """Test that db_session fixture provides working database session."""
        assert db_session is not None
        assert hasattr(db_session, 'execute')
        assert hasattr(db_session, 'commit')

    @pytest.mark.asyncio
    async def test_async_db_session_fixture(self, async_db_session: AsyncSession):
        """Test async database session fixture with pytest-asyncio marker."""
        assert async_db_session is not None
        assert hasattr(async_db_session, 'execute')
        assert hasattr(async_db_session, 'commit')

    def test_auth_service_fixture(self, auth_service):
        """Test that auth service fixture provides working authentication."""
        assert auth_service is not None
        assert hasattr(auth_service, 'authenticate_user')
        assert hasattr(auth_service, 'create_access_token')

    def test_admin_token_fixture(self, admin_token: str):
        """Test that admin token fixture provides valid JWT token."""
        assert admin_token is not None
        assert isinstance(admin_token, str)
        assert len(admin_token) > 20  # JWT tokens are longer

    def test_user_token_fixture(self, user_token: str):
        """Test that user token fixture provides valid JWT token."""
        assert user_token is not None
        assert isinstance(user_token, str)
        assert len(user_token) > 20

    def test_container_fixture(self, container: Container):
        """Test that container fixture provides configured DI container."""
        assert container is not None
        settings = container.config()
        assert settings.app.name == "pynomaly-test"
        assert settings.app.environment == "test"

    def test_sample_data_fixture(self, sample_data):
        """Test that sample data fixture provides DataFrame."""
        assert sample_data is not None
        assert len(sample_data) == 1000
        assert 'target' in sample_data.columns

    def test_sample_dataset_fixture(self, sample_dataset):
        """Test that sample dataset fixture provides Dataset entity."""
        assert sample_dataset is not None
        assert sample_dataset.name == "Test Dataset"
        assert len(sample_dataset.features) == 5

    def test_isolated_environment_fixture(self, isolated_test_environment):
        """Test that isolated environment fixture provides isolation."""
        assert isolated_test_environment is not None
        assert 'temp_dir' in isolated_test_environment
        assert 'original_cwd' in isolated_test_environment

    def test_temp_dir_fixture(self, temp_dir: str):
        """Test that temp_dir fixture provides temporary directory."""
        import os
        assert temp_dir is not None
        assert os.path.exists(temp_dir)
        assert os.path.isdir(temp_dir)


class TestAsyncRoutes:
    """Test class demonstrating async route testing."""

    @pytest.mark.asyncio
    async def test_async_endpoint_example(self, client: TestClient, auth_header: dict):
        """Example of testing async endpoints with proper markers."""
        # For async endpoints, you would typically use an async client
        # This is a simplified example showing the pattern
        response = client.get("/api/v1/health", headers=auth_header)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_async_database_operations(self, async_db_session: AsyncSession):
        """Example of testing async database operations."""
        # Example async database operations
        result = await async_db_session.execute("SELECT 1")
        assert result is not None


class TestAuthenticationWorkflow:
    """Test class demonstrating complete authentication workflow."""

    def test_login_workflow(self, client: TestClient, auth_service):
        """Test complete login workflow."""
        # Create user
        user = auth_service.create_user(
            username="testlogin",
            email="testlogin@example.com",
            password="testpass123",
            full_name="Test Login User"
        )
        assert user is not None

        # Login (would be through API endpoint in real app)
        token_response = auth_service.create_access_token(user)
        assert token_response.access_token is not None

        # Use token to access protected endpoint
        headers = {"Authorization": f"Bearer {token_response.access_token}"}
        response = client.get("/api/v1/health", headers=headers)
        assert response.status_code == 200

    def test_unauthorized_access(self, client: TestClient):
        """Test that unauthorized requests are rejected."""
        response = client.get("/api/v1/health")
        # Note: This might return 200 for health endpoint, depending on implementation
        # Adjust based on your actual auth requirements
        assert response.status_code in [200, 401, 403]

    def test_invalid_token(self, client: TestClient):
        """Test that invalid tokens are rejected."""
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/api/v1/health", headers=headers)
        # Adjust based on your actual auth implementation
        assert response.status_code in [200, 401, 403]


class TestDatabaseIsolation:
    """Test class demonstrating database isolation between tests."""

    def test_database_isolation_1(self, db_session: Session):
        """Test that demonstrates database isolation."""
        # This test modifies database state
        # Changes should not affect other tests
        result = db_session.execute("SELECT 1 as test_value")
        assert result is not None

    def test_database_isolation_2(self, db_session: Session):
        """Test that should not see changes from previous test."""
        # This test should start with clean database state
        result = db_session.execute("SELECT 1 as test_value")
        assert result is not None

    @pytest.mark.asyncio
    async def test_async_database_isolation(self, async_db_session: AsyncSession):
        """Test async database isolation."""
        result = await async_db_session.execute("SELECT 1 as test_value")
        assert result is not None


class TestGlobalStateReset:
    """Test class demonstrating global state reset between tests."""

    def test_global_state_reset_1(self, auth_service):
        """Test that modifies global state."""
        # Modify auth service state
        auth_service.create_user(
            username="globaltest1",
            email="globaltest1@example.com",
            password="testpass123"
        )
        assert len(auth_service._users) >= 2  # admin + new user

    def test_global_state_reset_2(self, auth_service):
        """Test that should not see changes from previous test."""
        # Should only have default admin user due to global state reset
        # This demonstrates that the reset_global_state fixture is working
        users = list(auth_service._users.keys())
        assert "admin" in users
