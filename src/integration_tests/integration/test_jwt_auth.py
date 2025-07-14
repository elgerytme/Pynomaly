import pytest
from fastapi.testclient import TestClient

from pynomaly.infrastructure.auth.jwt_auth import JWTAuthService, get_auth, init_auth
from pynomaly.infrastructure.config import get_settings
from pynomaly.presentation.api.app import create_app


@pytest.fixture(scope="module")
def app():
    settings = get_settings()
    auth_service = init_auth(settings)
    app = create_app()
    app.dependency_overrides[get_auth] = lambda: auth_service
    return app


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app)


def test_login(client):
    response = client.post("/api/v1/auth/login", data={"username": "admin", "password": "admin123"})
    assert response.status_code == 200
    json_response = response.json()
    assert "access_token" in json_response


def test_login_with_invalid_credentials(client):
    response = client.post("/api/v1/auth/login", data={"username": "admin", "password": "wrongpassword"})
    assert response.status_code == 401


def test_login_locked_out(client):
    # Attempt failed logins to trigger account lockout
    for _ in range(6):
        client.post("/api/v1/auth/login", data={"username": "admin", "password": "wrongpassword"})

    response = client.post("/api/v1/auth/login", data={"username": "admin", "password": "admin123"})
    assert response.status_code == 401
    assert response.json().get("detail") == "Account temporarily locked due to too many failed login attempts"


def test_password_change(client):
    # Log in with correct details to reset failed login attempts
    response = client.post("/api/v1/auth/login", data={"username": "admin", "password": "admin123"})
    assert response.status_code == 200
    json_response = response.json()

    # Change password
    auth_service: JWTAuthService = get_auth()
    success = auth_service.change_password(user_id="admin", old_password="admin123", new_password="newpassword123")
    assert success is True

    # Ensure login works with new password
    response = client.post("/api/v1/auth/login", data={"username": "admin", "password": "newpassword123"})
    assert response.status_code == 200


def test_refresh_token(client):
    response = client.post("/api/v1/auth/login", data={"username": "admin", "password": "newpassword123"})
    assert response.status_code == 200
    json_response = response.json()
    refresh_token = json_response.get("refresh_token")

    response = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert response.status_code == 200
    assert "access_token" in response.json()
