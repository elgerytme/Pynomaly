import json
import os
import pytest
from fastapi.testclient import TestClient
from pynomaly.presentation.api.app import create_app
from pynomaly.infrastructure.config import create_container


@pytest.fixture
def app():
    """Create app with test container."""
    container = create_container(testing=True)
    return create_app(container=container)


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def token(client):
    """Obtain an authentication token for testing protected endpoints."""
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_create_read_update_delete_dataset(token):
    """Test CRUD operations for a dataset."""
    headers = {"Authorization": f"Bearer {token}"}

    # Create dataset
    response = client.post(
        "/api/v1/datasets/",
        files={"file": ("test.csv", "id,value\n1,100\n2,200", "text/csv")},
        headers=headers,
    )
    assert response.status_code == 201
    dataset_id = response.json()["id"]

    # Read dataset
    response = client.get(f"/api/v1/datasets/{dataset_id}", headers=headers)
    assert response.status_code == 200

    # Update dataset
    response = client.put(
        f"/api/v1/datasets/{dataset_id}",
        json={"name": "Updated Dataset"},
        headers=headers,
    )
    assert response.status_code == 200

    # Delete dataset
    response = client.delete(f"/api/v1/datasets/{dataset_id}", headers=headers)
    assert response.status_code == 204


def test_file_upload_with_auth(token):
    """Test file uploading with proper authentication."""
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post(
        "/api/v1/files/upload",
        files={"file": ("test.csv", "id,value\n1,100\n2,200", "text/csv")},
        headers=headers,
    )
    assert response.status_code == 201


# def test_streaming_sse_or_websockets():
#     """Test streaming endpoints if applicable."""
#     # This example is for SSE, adjust as needed for websockets
#     response = client.get("/api/v1/streaming-endpoint")
#     assert response.status_code == 200
#     assert "text/event-stream" in response.headers["content-type"]


def test_jwt_auth_flows():
    """Test JWT authentication flows."""
    response = client.post(
        "/api/v1/auth/login",
        data=json.dumps({"username": "admin", "password": "admin123"}),
    )
    assert response.status_code == 200
    token = response.json()["access_token"]

    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/api/v1/auth/me", headers=headers)
    assert response.status_code == 200

    # Test logout
    response = client.post("/api/v1/auth/logout", headers=headers)
    assert response.status_code == 200


def test_openapi_schema_validation():
    """Validate OpenAPI schema generation."""
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    assert "paths" in response.json()

