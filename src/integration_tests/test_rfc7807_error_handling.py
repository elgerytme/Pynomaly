"""Tests for RFC 7807 Problem Details error handling."""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from pynomaly.infrastructure.error_handling.problem_details_handler import (
    ProblemDetailsResponse,
    add_exception_handlers,
    create_problem_details_response,
)
from pynomaly.infrastructure.middleware.correlation_id_middleware import (
    add_correlation_id,
)


class SampleModel(BaseModel):
    """Test model with strict validation."""

    model_config = {"extra": "forbid"}

    name: str
    age: int


def test_problem_details_response_structure():
    """Test that ProblemDetailsResponse follows RFC 7807 structure."""
    response = ProblemDetailsResponse(
        status_code=400,
        title="Bad Request",
        detail="Invalid input provided",
        instance="/api/v1/test",
        correlation_id="test-123",
    )

    assert response.status_code == 400
    assert response.headers["X-Correlation-ID"] == "test-123"

    # Check content structure
    content = response.body.decode()
    assert "type" in content
    assert "title" in content
    assert "status" in content
    assert "detail" in content
    assert "instance" in content


def test_validation_error_handling():
    """Test that validation errors are handled correctly."""
    app = FastAPI()
    add_exception_handlers(app)

    @app.post("/test")
    async def test_endpoint(data: SampleModel):
        return {"message": "success"}

    client = TestClient(app)

    # Test invalid data
    response = client.post("/test", json={"name": "test", "age": "not_an_int"})
    assert response.status_code == 422

    data = response.json()
    assert data["type"] == "https://tools.ietf.org/html/rfc7231#section-6.5.1"
    assert data["title"] == "Validation Error"
    assert data["status"] == 422
    assert "validation_errors" in data


def test_strict_validation_forbids_extra_fields():
    """Test that extra fields are rejected with strict validation."""
    app = FastAPI()
    add_exception_handlers(app)

    @app.post("/test")
    async def test_endpoint(data: SampleModel):
        return {"message": "success"}

    client = TestClient(app)

    # Test with extra field
    response = client.post(
        "/test", json={"name": "test", "age": 25, "extra_field": "should_be_rejected"}
    )
    assert response.status_code == 422

    data = response.json()
    assert data["title"] == "Validation Error"
    assert "extra_field" in data["detail"]


def test_http_exception_handling():
    """Test that HTTP exceptions are handled correctly."""
    app = FastAPI()
    add_exception_handlers(app)

    @app.get("/test")
    async def test_endpoint():
        raise HTTPException(status_code=404, detail="Resource not found")

    client = TestClient(app)

    response = client.get("/test")
    assert response.status_code == 404

    data = response.json()
    assert data["type"] == "about:blank"
    assert data["title"] == "Not Found"
    assert data["status"] == 404
    assert data["detail"] == "Resource not found"


def test_correlation_id_middleware():
    """Test that correlation IDs are properly handled."""
    app = FastAPI()
    app.middleware("http")(add_correlation_id)
    add_exception_handlers(app)

    @app.get("/test")
    async def test_endpoint():
        return {"message": "success"}

    @app.get("/error")
    async def error_endpoint():
        raise HTTPException(status_code=500, detail="Server error")

    client = TestClient(app)

    # Test successful request
    response = client.get("/test")
    assert "X-Correlation-ID" in response.headers

    # Test error response includes correlation ID
    response = client.get("/error")
    assert "X-Correlation-ID" in response.headers

    # Test with custom correlation ID
    custom_id = "custom-123"
    response = client.get("/test", headers={"X-Correlation-ID": custom_id})
    assert response.headers["X-Correlation-ID"] == custom_id


def test_general_exception_handling():
    """Test that general exceptions are handled correctly."""
    app = FastAPI()
    add_exception_handlers(app)

    @app.get("/test")
    async def test_endpoint():
        raise ValueError("Something went wrong")

    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/test")
    assert response.status_code == 500

    data = response.json()
    assert data["type"] == "about:blank"
    assert data["title"] == "Internal Server Error"
    assert data["status"] == 500
    assert data["detail"] == "An unexpected error occurred. Please try again later."


def test_create_problem_details_response():
    """Test the utility function for creating problem details responses."""
    from unittest.mock import Mock

    request = Mock()
    request.url = "http://example.com/test"
    request.state.correlation_id = "test-123"

    response = create_problem_details_response(
        request=request,
        status_code=400,
        title="Bad Request",
        detail="Invalid input",
        extensions={"error_code": "INVALID_INPUT"},
    )

    assert response.status_code == 400
    assert response.headers["X-Correlation-ID"] == "test-123"


def test_dto_extra_forbid():
    """Test that DTOs reject extra fields."""
    from uuid import uuid4

    from pynomaly.application.dto.detection_dto import DetectionRequestDTO

    # Valid data should work
    valid_data = {
        "detector_id": str(uuid4()),
        "dataset_id": str(uuid4()),
        "threshold": 0.5,
    }
    dto = DetectionRequestDTO(**valid_data)
    assert dto.detector_id is not None

    # Extra fields should be rejected
    invalid_data = {
        "detector_id": str(uuid4()),
        "dataset_id": str(uuid4()),
        "threshold": 0.5,
        "extra_field": "should_be_rejected",
    }

    with pytest.raises(ValidationError) as exc_info:
        DetectionRequestDTO(**invalid_data)

    assert "extra_field" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
